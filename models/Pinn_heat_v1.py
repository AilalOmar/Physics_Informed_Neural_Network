import tensorflow as tf
import numpy as np
import time

# Set TensorFlow logging to only show errors
tf.get_logger().setLevel('ERROR')

# --- 1. Problem Definition and Domain Setup ---

# Physical parameters and domain
ALPHA = 0.01   # Thermal diffusivity
L = 1.0        # Spatial domain length [0, L]
T_MAX = 1.0    # Maximum time domain [0, T_MAX]

# --- 2. Initial and Boundary Conditions ---

# Initial Condition (IC): u(x, 0) = sin(pi * x)
def initial_condition(x):
    """The temperature distribution at t=0."""
    return np.sin(np.pi * x)

# Boundary Condition (BC): u(0, t) = 0 and u(L, t) = 0 (Dirichlet boundary)
# The NN is simply trained to predict 0 at these boundary points.
# Since the BC is constant (zero), the target value is simply 0.


# --- 3. The PINN Model Architecture ---

def create_pinn_model():
    """Constructs the neural network model."""
    # Input: (x, t) coordinates
    input_layer = tf.keras.Input(shape=(2,))
    
    # 5 Hidden Layers, 20 neurons per layer, using tanh activation (common in PINNs)
    x = input_layer
    for i in range(5):
        x = tf.keras.layers.Dense(20, activation='tanh', 
                                  kernel_initializer='glorot_normal', 
                                  name=f'h_layer_{i+1}')(x)
        
    # Output: u(x, t)
    output_layer = tf.keras.layers.Dense(1, name='output')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the model instance
model = create_pinn_model()
model.summary()


# --- 4. Custom Training Loop and Loss Function ---

# PINNs require a custom training method to calculate the physics loss (L_R)
# using automatic differentiation.

@tf.function
def pinn_loss(X_R, X_IC, U_IC, X_BC, U_BC):
    """
    Calculates the total PINN loss: L_Total = L_R + L_IC + L_BC.
    
    Args:
        X_R, X_IC, X_BC: Tensors of collocation points (x, t) for Residual, 
                         Initial Condition, and Boundary Condition, respectively.
        U_IC, U_BC: Known target values for IC and BC points.
    """
    # --- 1. Physics (Residual) Loss L_R ---
    with tf.GradientTape(persistent=True) as tape:
        # The variables x and t are the first and second columns of X_R
        x = X_R[:, 0:1]
        t = X_R[:, 1:2]
        
        # Watch the input variables required for differentiation
        tape.watch([x, t])
        
        # Predict u_NN(x, t)
        X_R_watched = tf.stack([x[:, 0], t[:, 0]], axis=1)
        u_NN = model(X_R_watched)

        # Calculate first derivative: d(u_NN)/dx and d(u_NN)/dt
        u_t = tape.gradient(u_NN, t)
        u_x = tape.gradient(u_NN, x)
        
        # Calculate second derivative: d^2(u_NN)/dx^2
        u_xx = tape.gradient(u_x, x)

    del tape # Cleanup tape resources

    # The residual R(x, t) = u_t - alpha * u_xx
    residual = u_t - ALPHA * u_xx
    L_R = tf.reduce_mean(tf.square(residual))

    # --- 2. Initial Condition Loss L_IC ---
    u_NN_IC = model(X_IC)
    L_IC = tf.reduce_mean(tf.square(u_NN_IC - U_IC))

    # --- 3. Boundary Condition Loss L_BC ---
    u_NN_BC = model(X_BC)
    L_BC = tf.reduce_mean(tf.square(u_NN_BC - U_BC))

    # Total loss (we use equal weighting here for simplicity)
    L_Total = L_R + L_IC + L_BC
    
    return L_Total, L_R, L_IC, L_BC


# --- 5. Data Generation (Training Points) ---

def generate_training_data(N_R, N_IC, N_BC):
    """Generates the collocation points for the three loss components."""
    
    # 5.1. Residual Points (X_R): Interior (x, t) points
    # Uniformly sample x in [0, L] and t in [0, T_MAX]
    x_R = np.random.uniform(0, L, N_R)
    t_R = np.random.uniform(0, T_MAX, N_R)
    X_R = np.vstack((x_R, t_R)).T.astype(np.float32)
    # Convert to Tensor and mark x, t as required for differentiation
    X_R = tf.convert_to_tensor(X_R)
    
    # 5.2. Initial Condition Points (X_IC): (x, 0)
    # Sample x along the initial boundary
    x_IC = np.linspace(0, L, N_IC, dtype=np.float32)
    t_IC = np.zeros(N_IC, dtype=np.float32)
    X_IC = np.vstack((x_IC, t_IC)).T
    # The true initial values U_IC = u(x, 0)
    U_IC = initial_condition(x_IC).reshape(-1, 1).astype(np.float32)
    
    # 5.3. Boundary Condition Points (X_BC): (0, t) and (L, t)
    # Sample t along the spatial boundaries
    t_BC = np.linspace(0, T_MAX, N_BC // 2, dtype=np.float32)
    
    # Left Boundary (x=0, t)
    x_BC_L = np.zeros_like(t_BC)
    X_BC_L = np.vstack((x_BC_L, t_BC)).T
    U_BC_L = np.zeros_like(x_BC_L).reshape(-1, 1) # Target value is 0

    # Right Boundary (x=L, t)
    x_BC_R = L * np.ones_like(t_BC)
    X_BC_R = np.vstack((x_BC_R, t_BC)).T
    U_BC_R = np.zeros_like(x_BC_R).reshape(-1, 1) # Target value is 0
    
    # Combine boundary points
    X_BC = np.concatenate([X_BC_L, X_BC_R], axis=0).astype(np.float32)
    U_BC = np.concatenate([U_BC_L, U_BC_R], axis=0).astype(np.float32)
    
    return X_R, X_IC, U_IC, X_BC, U_BC

# --- 6. Training Execution ---

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Training hyper-parameters
N_R = 10000      # Number of Residual (collocation) points
N_IC = 100       # Number of Initial Condition points
N_BC = 100       # Number of Boundary Condition points (50 on left, 50 on right)
EPOCHS = 5000    # Number of training epochs

# Generate the data once
X_R, X_IC, U_IC, X_BC, U_BC = generate_training_data(N_R, N_IC, N_BC)

@tf.function
def train_step():
    """Performs a single optimization step."""
    with tf.GradientTape() as tape:
        L_Total, L_R, L_IC, L_BC = pinn_loss(X_R, X_IC, U_IC, X_BC, U_BC)
    
    # Compute gradients and apply updates
    gradients = tape.gradient(L_Total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return L_Total, L_R, L_IC, L_BC


print(f"\n--- Starting PINN Training for {EPOCHS} Epochs ---")
start_time = time.time()

# Main training loop
for epoch in range(1, EPOCHS + 1):
    L_Total, L_R, L_IC, L_BC = train_step()
    
    if epoch % 500 == 0 or epoch == 1:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:05d} | Time: {elapsed:.2f}s | "
              f"Total Loss: {L_Total.numpy():.4e} | "
              f"Physics Loss (L_R): {L_R.numpy():.4e} | "
              f"IC Loss (L_IC): {L_IC.numpy():.4e} | "
              f"BC Loss (L_BC): {L_BC.numpy():.4e}")

print("--- Training Complete ---")

# --- 7. Evaluation (Simple Prediction Grid) ---

# Create a grid of points for visualization
X_TEST = np.linspace(0, L, 100)
T_TEST = np.linspace(0, T_MAX, 100)
T_GRID, X_GRID = np.meshgrid(T_TEST, X_TEST)

# Flatten and stack the grid points for prediction
X_T_FLAT = np.vstack([X_GRID.flatten(), T_GRID.flatten()]).T.astype(np.float32)

# Predict the temperature u(x, t) using the trained model
U_PREDICTED = model.predict(X_T_FLAT)

print(f"\nExample Prediction (x=0.5, t=0.1): "
      f"u({0.5}, {0.1}) = {model(np.array([[0.5, 0.1]])).numpy()[0, 0]:.4f}")
print(f"Example Prediction (x=0.5, t=0.5): "
      f"u({0.5}, {0.5}) = {model(np.array([[0.5, 0.5]])).numpy()[0, 0]:.4f}")

# Note: The output U_PREDICTED can be reshaped to (100, 100) for plotting a contour map.
# U_GRID = U_PREDICTED.reshape(100, 100)
import matplotlib.pyplot as plt
from matplotlib.cm import jet # Pour une palette de couleurs dynamique

# --- Visualisation de la solution PINN ---

# Les données X_GRID, T_GRID et U_PREDICTED sont déjà calculées à la fin du code

# 1. Reshape la prédiction pour correspondre à la grille
# X_GRID et T_GRID ont la forme (100, 100). U_PREDICTED est (10000, 1).
U_GRID = U_PREDICTED.reshape(X_GRID.shape)

# Crée la figure et les axes
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Solution de l'équation de la chaleur 1D par PINN", fontsize=16)

# --- A. Carte de Contour (Heatmap) ---
# Afficher l'évolution de la température u(x, t) sur le domaine x-t
im = ax[0].contourf(T_GRID, X_GRID, U_GRID, 100, cmap=jet) # 100 niveaux de contour
fig.colorbar(im, ax=ax[0], label="Température u(x, t)")

ax[0].set_xlabel("Temps t")
ax[0].set_ylabel("Position x")
ax[0].set_title("Distribution de la Température u(x, t) (Carte de Contour)")

# --- B. Profils de Température à différents instants t ---
# Montrer comment la température u(x) change avec le temps

# Sélectionner les indices pour t = 0 (IC), t = 0.25, t = 0.5, t = 1.0 (T_MAX)
# Indices t_test : 0 (t=0), 25 (t=0.25), 50 (t=0.5), 99 (t=1.0)
time_indices = [0, 25, 50, 99]
time_values = T_TEST[time_indices] # Récupérer les vraies valeurs de temps

for i, t_idx in enumerate(time_indices):
    # U_GRID[i, :] correspond à x, t_i
    ax[1].plot(X_TEST, U_GRID[:, t_idx], label=f"t = {time_values[i]:.2f}")

ax[1].set_xlabel("Position x")
ax[1].set_ylabel("Température u(x, t)")
ax[1].set_title("Profils de Température u(x) à des instants t Fixés")
ax[1].legend()
ax[1].grid(True, linestyle='--')

plt.tight_layout()
plt.show()