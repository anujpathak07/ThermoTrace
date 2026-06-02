# =============================================================================
# ThermoTrace — Configuration
# Inverse Heat Source Identification using Machine Learning
# =============================================================================

# --- Physical Domain ---
PLATE_SIZE_M     = 0.1          # 100mm × 100mm plate (meters)
GRID_SIZE        = 64           # 64×64 discretization
T_BOUNDARY       = 300.0        # Fixed boundary temperature (K)
K_THERMAL        = 50.0         # Thermal conductivity of steel (W/m·K)

# --- Heat Source Parameters ---
MIN_SOURCES      = 1
MAX_SOURCES      = 3
MIN_INTENSITY    = 1e4          # W/m³
MAX_INTENSITY    = 5e5          # W/m³
MIN_RADIUS_CELLS = 3            # Minimum source radius in grid cells
MAX_RADIUS_CELLS = 7            # Maximum source radius in grid cells
SOURCE_MARGIN    = 10           # Minimum distance from boundary (cells)

# --- Dataset ---
NUM_TRAIN        = 8000
NUM_VAL          = 1000
NUM_TEST         = 1000
TOTAL_SAMPLES    = NUM_TRAIN + NUM_VAL + NUM_TEST
DATA_DIR         = "data"
TRAIN_FILE       = "train.npz"
VAL_FILE         = "val.npz"
TEST_FILE        = "test.npz"

# --- Noise (for publication: simulate real sensor noise) ---
ADD_NOISE        = True
NOISE_STD_K      = 0.1          # Gaussian noise std dev in Kelvin (realistic thermocouple noise)

# --- Model ---
MODEL_DIR        = "checkpoints"
RESULTS_DIR      = "results"

# --- Training ---
BATCH_SIZE       = 32
LEARNING_RATE    = 1e-3
NUM_EPOCHS       = 100
PATIENCE         = 15           # Early stopping patience
