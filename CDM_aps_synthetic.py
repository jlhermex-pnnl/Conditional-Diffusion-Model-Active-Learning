import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import sys

# Configure matplotlib for non-blocking plots
#matplotlib.use('TkAgg')  # Use TkAgg backend for non-blocking behavior
plt.ion()  # Turn on interactive mode
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import MDS
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, random_split, Dataset
import pflotran_tank_sim as pf  
from torch.utils.data import ConcatDataset
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#%% Help Functions and Classes

# Fix random seeds for reproducibility
def set_random_seeds(seed=2054):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

# Configuration Parameters
class Config:
    DATA_DIR = './GIFT_Data_Package/GIFT_uniform_synthetic'
    LEARNING_STAGE_DIR = './Inverted_field_simulations'
    if not os.path.exists(LEARNING_STAGE_DIR):
        os.makedirs(LEARNING_STAGE_DIR)
    SIM_OUT_DIR = './Sim_out_inverted_field_simulations'
    if not os.path.exists(SIM_OUT_DIR):
        os.makedirs(SIM_OUT_DIR)
    PFTEMPLATE = 'gift_pflotran_template.in'
    CHECKPOINT_DIR = './checkpoints'
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    RESULTS_DIR = './GIFT_CDM_Results'
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    BATCH_SIZE = 8
    NUM_EPOCHS = 500
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_LEARNING_STAGES = 200
    RESUME_FROM_STAGE = None  # Set to stage number to resume from that stage, or None to start fresh
    MAX_AUGMENTED_SAMPLE_ATTEMPTS = 500
    ALPHA_SD = 0.1
    BETA_SD = 0.01
    TDIFFUSION_TIMESTEPS = 500
    NUM_SAMPLES = 30  # Number of CDM samples (will be stage-dependent)
    NUM_UNCERTAINTY_SAMPLES = 30  # Number of uncertainty-based samples (will be stage-dependent)
    FIELD_EVAL_SAMPLE_COUNT = 5  # PFLOTRAN evaluations to track CDM field fit
    PARAMETER_NAMES = [
        '30 Sand Archie Cementation_exponent',
        '30 Sand Arch Tortuosity_constant',
        '30 Sand Archie Surface_EC',
        '30 Sand Porosity',
        '30 Sand Permeability',
        '70 Sand Archie Cementation_exponent',
        '70 Sand Arch Tortuosity_constant',
        '70 Sand Archie Surface_EC',
        '70 Sand Porosity',
        '70 Sand Permeability',
        '16 Sand Archie Cementation_exponent',
        '16 Sand Arch Tortuosity_constant',
        '16 Sand Archie Surface EC',
        '16 Sand Porosity',
        '16 Sand Permeability'
    ]
    LOG_PARAMETERS = [2, 4, 7, 9, 12, 14]  # Parameters known to be log-ed


# Data Loading and Preprocessing Utilities
class DataLoaderHandler:
    """
    Handles data loading, scaling, and preprocessing with consistent scaler management.
    Args:
        config: Configuration object with data directory and other settings
    """
    def __init__(self, config: Config):
        self.config = config
        self.parameter_scalers = {}


    def load_data(self, total_simulations=300, stage=None, case_name='test'):
        """
        Unified data loading function for both original training data and augmented simulation data.
        
        Args:
            total_simulations: Number of simulations to load
            stage: Learning stage identifier. If None, loads test case data.
                  If string/int, loads augmented data from that stage.
            case_name: Simulation name prefix ('Test', 'WSSE_guided', or 'Active_CDM')
                  
        Returns:
            tuple: (ertt_data, ertd_data, parameter_data_array, conductivity_data, tracer_data)
        """
        ertt_data, ertd_data, cond_data, parameter_data = [], [], [], []
        
        data_dir = self.config.LEARNING_STAGE_DIR
        
        # Handle different file naming patterns based on stage and case_name
        if stage is None:
            # Test case: stage=None means field parameter verification
            # File pattern: {tag}_{i}_Test.npy
            file_suffix = f"_{case_name}"
            print(f"Loading test simulation data from: '{data_dir}'")
            print(f"  File pattern: {{tag}}_{{i}}{file_suffix}.npy")
        else:
            # Active learning stage: use case_name in the naming pattern
            # File pattern: {tag}_{case_name}_{stage}_{i}_stage_{stage}.npy
            file_suffix = f"_stage_{stage}"
            print(f"Loading {case_name} simulation data from stage {stage}: '{data_dir}'")
            print(f"  File pattern: {{tag}}_{case_name}_{stage}_{{i}}{file_suffix}.npy")

        for i in range(total_simulations):
            try:
                if stage is None:
                    # Test case file names: {tag}_{i}_Test.npy
                    ertt_data.append(np.load(f"{data_dir}/ertt_{i}{file_suffix}.npy"))
                    ertd_data.append(np.load(f"{data_dir}/ertd_{i}{file_suffix}.npy"))
                    cond_data.append(np.load(f"{data_dir}/obsd_{i}{file_suffix}.npy"))
                    parameter_data.append(np.load(f"{data_dir}/params_{i}{file_suffix}.npy"))
                else:
                    # Active learning stage file names: {tag}_{case_name}_{stage}_{i}_stage_{stage}.npy
                    ertt_data.append(np.load(f"{data_dir}/ertt_{case_name}_{stage}_{i}{file_suffix}.npy"))
                    ertd_data.append(np.load(f"{data_dir}/ertd_{case_name}_{stage}_{i}{file_suffix}.npy"))
                    cond_data.append(np.load(f"{data_dir}/obsd_{case_name}_{stage}_{i}{file_suffix}.npy"))
                    parameter_data.append(np.load(f"{data_dir}/params_{case_name}_{stage}_{i}{file_suffix}.npy"))
            except FileNotFoundError as e:
                print(f"Warning: Could not load file for simulation {i}: {e}")
                continue

        if not ertt_data:
            raise ValueError(f"No data files found. Check directory: {data_dir}")

        print(f"  Successfully loaded {len(ertt_data)} simulations")
        
        ertt_data = np.array(ertt_data)
        ertd_data = np.array(ertd_data).swapaxes(1, 2)
        cond_data = np.array(cond_data)
        parameter_data_array = np.array(parameter_data)
        
        # Extract conductivity and tracer data from combined observation data
        conductivity_data = cond_data[:, 0, :, :]
        tracer_cont_data = cond_data[:, 1, :, :]

        return ertt_data, ertd_data, parameter_data_array, conductivity_data, tracer_cont_data

    def load_data_from_npz(self, npz_file_path):
        """
        Load data from a pre-saved NPZ file.
        Args:
            npz_file_path: Path to the NPZ file containing the data 
        Returns:
            tuple: (ertt_data, ertd_data, parameter_data_array, conductivity_data, tracer_cont_data)
        """
        print(f"Loading data from NPZ file: {npz_file_path}")
        
        if not os.path.exists(npz_file_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_file_path}")
        
        try:
            # Load the NPZ file
            data = np.load(npz_file_path)
            
            # Extract the data using the keys you specified
            ertt_data = data['ertt']
            ertd_data = data['ertd']
            parameter_data_array = data['params']
            conductivity_data = data['conductivity']
            tracer_cont_data = data['tracer']
            
            print(f"Successfully loaded data from NPZ:")
            print(f"  ERT-T data shape: {ertt_data.shape}")
            print(f"  ERT-D data shape: {ertd_data.shape}")
            print(f"  Parameter data shape: {parameter_data_array.shape}")
            print(f"  Conductivity data shape: {conductivity_data.shape}")
            print(f"  Tracer data shape: {tracer_cont_data.shape}")
            
            return ertt_data, ertd_data, parameter_data_array, conductivity_data, tracer_cont_data
            
        except KeyError as e:
            print(f"Error: Missing key in NPZ file: {e}")
            print(f"Available keys in the file: {list(data.keys())}")
            raise
        except Exception as e:
            print(f"Error loading NPZ file: {e}")
            raise

    def load_data_auto(self, npz_file_path=None, total_simulations=293, stage=None):
        """
        Automatically choose between NPZ loading and individual file loading.
        Args:
            npz_file_path: Path to NPZ file. If provided and exists, loads from NPZ.
            total_simulations: Number of simulations to load (for individual files)
            stage: Learning stage identifier (for individual files)
            
        Returns:
            tuple: (ertt_data, ertd_data, parameter_data_array, conductivity_data, tracer_cont_data)
        """
        if npz_file_path and os.path.exists(npz_file_path):
            print(f"NPZ file found: {npz_file_path}")
            print("Loading data from NPZ file...")
            return self.load_data_from_npz(npz_file_path)
        else:
            if npz_file_path:
                print(f"NPZ file not found: {npz_file_path}")
            print("Falling back to individual file loading...")
            return self.load_data(total_simulations=total_simulations, stage=stage)

    def establish_scalers(self, parameter_data_array, ert_array, tracer_array, cond_array):  
        """
        Establish and store all scalers for consistent normalization across the pipeline.
        Args:
            parameter_data_array: Parameter data for establishing parameter scalers
            ert_array: ERT data for establishing ERT scaler
            tracer_array: Tracer data for establishing tracer scaler
            cond_array: Conductivity data for establishing conductivity scaler
        """
        print("Establishing all data scalers...")
        
        # Store reference shape for ERT data consistency checking
        self._reference_ert_shape = ert_array.shape
        
        # Establish parameter scalers (MinMaxScaler for each parameter)
        # Apply log transformation to log parameters before fitting scaler
        self.parameter_scalers = {}
        for i in range(parameter_data_array.shape[1]):
            scaler = MinMaxScaler()
            
            # Apply log transformation if this is a log parameter
            if i in self.config.LOG_PARAMETERS:
                param_data = np.log10(parameter_data_array[:, i])
                print(f"Parameter {i+1} (LOG) scaler established - Original Range: [{parameter_data_array[:, i].min():.4f}, {parameter_data_array[:, i].max():.4f}]")
                print(f"                           Log10 Range: [{param_data.min():.4f}, {param_data.max():.4f}]")
            else:
                param_data = parameter_data_array[:, i]
                print(f"Parameter {i+1} scaler established - Range: [{param_data.min():.4f}, {param_data.max():.4f}]")
            
            scaler.fit(param_data.reshape(-1, 1))
            self.parameter_scalers[i] = scaler

        # Establish ERT scaler (StandardScaler)
        self.ert_scaler = StandardScaler()
        self.ert_scaler.fit(ert_array.reshape(-1, 1))
        print(f"ERT scaler established - Shape: {ert_array.shape}")

        # Establish tracer scaler (StandardScaler)
        self.tracer_scaler = StandardScaler()
        self.tracer_scaler.fit(tracer_array.reshape(-1, 1))
        print(f"Tracer scaler established - Shape: {tracer_array.shape}")

        # Establish conductivity scaler (StandardScaler)
        self.conductivity_scaler = StandardScaler()
        self.conductivity_scaler.fit(cond_array.reshape(-1, 1))
        print(f"Conductivity scaler established - Shape: {cond_array.shape}")
        
        print("All scalers established successfully!")

    def normalize_parameters(self, parameter_data_array):
        """
        Normalize parameters using pre-established MinMaxScalers.
        Args:
            parameter_data_array: Parameter data to normalize
        Returns:
            parameters_norm: Normalized parameter data
        """
        if not hasattr(self, 'parameter_scalers') or not self.parameter_scalers:
            raise ValueError("Parameter scalers not established. Call establish_scalers() first.")
            
        parameters_norm = np.zeros_like(parameter_data_array)
        
        for i in range(parameter_data_array.shape[1]):
            if i in self.config.LOG_PARAMETERS:
                param_data = np.log10(parameter_data_array[:, i])
            else:
                param_data = parameter_data_array[:, i]
            
            parameters_norm[:, i] = self.parameter_scalers[i].transform(
                param_data.reshape(-1, 1)
            ).flatten()
            
        return parameters_norm

    def scale_ert_data(self, ert_data):
        """
        Scale ERT data using pre-established scaler.
        Args:
            ert_data: ERT data to scale
        Returns:
            scaled_data: Scaled ERT data
        """
        if not hasattr(self, 'ert_scaler'):
            raise ValueError("ERT scaler not established. Call establish_scalers() first.")
            
        scaled_data = self.ert_scaler.transform(ert_data.reshape(-1, 1)).reshape(ert_data.shape)
        return scaled_data

    def scale_tracer_data(self, tracer_data):
        """
        Scale tracer data using pre-established scaler.
        Args:
            tracer_data: Tracer data to scale
        Returns:
            scaled_data: Scaled tracer data
        """
        if not hasattr(self, 'tracer_scaler'):
            raise ValueError("Tracer scaler not established. Call establish_scalers() first.")
            
        scaled_data = self.tracer_scaler.transform(tracer_data.reshape(-1, 1)).reshape(tracer_data.shape)
        return scaled_data

    def scale_conductivity_data(self, conductivity_data):
        """
        Scale conductivity data using pre-established scaler.
        Args:
            conductivity_data: Conductivity data to scale
        Returns:
            scaled_data: Scaled conductivity data
        """
        if not hasattr(self, 'conductivity_scaler'):
            raise ValueError("Conductivity scaler not established. Call establish_scalers() first.")
            
        scaled_data = self.conductivity_scaler.transform(conductivity_data.reshape(-1, 1)).reshape(conductivity_data.shape)
        return scaled_data

    def inverse_transform_parameters(self, scaled_params):
        """
        Inverse transform scaled parameters back to original scale using stored scalers.
        Applies inverse log10 transformation (10^x) to log parameters after denormalization.
        Args:
            scaled_params: Normalized parameters (1D or 2D array)
        Returns:
            params_original: Parameters in original scale
        """
        if not hasattr(self, 'parameter_scalers') or not self.parameter_scalers:
            raise ValueError("Parameter scalers not established. Call establish_scalers() first.")
            
        if scaled_params.ndim == 1:  # Single sample
            params_original = np.zeros_like(scaled_params)
            for i in range(len(scaled_params)):
                # Inverse transform from normalized space
                param_value = self.parameter_scalers[i].inverse_transform([[scaled_params[i]]])[0, 0]
                
                # Apply inverse log transformation if this is a log parameter
                if i in self.config.LOG_PARAMETERS:
                    params_original[i] = 10 ** param_value  # 10^x to reverse log10
                else:
                    params_original[i] = param_value
                    
        else:  # Multiple samples
            params_original = np.zeros_like(scaled_params)
            for i in range(scaled_params.shape[1]):
                # Inverse transform from normalized space
                param_values = self.parameter_scalers[i].inverse_transform(scaled_params[:, i].reshape(-1, 1)).flatten()
                
                # Apply inverse log transformation if this is a log parameter
                if i in self.config.LOG_PARAMETERS:
                    params_original[:, i] = 10 ** param_values  # 10^x to reverse log10
                else:
                    params_original[:, i] = param_values
                    
        return params_original

    def inverse_transform_ert_data(self, scaled_ert_data):
        """
        Inverse transform scaled ERT data back to original scale.
        Args:
            scaled_ert_data: Scaled ERT data
        Returns:
            ert_original: ERT data in original scale
        """
        if not hasattr(self, 'ert_scaler'):
            raise ValueError("ERT scaler not established. Call establish_scalers() first.")
            
        original_shape = scaled_ert_data.shape
        ert_original = self.ert_scaler.inverse_transform(scaled_ert_data.reshape(-1, 1)).reshape(original_shape)
        return ert_original

    def inverse_transform_tracer_data(self, scaled_tracer_data):
        """
        Inverse transform scaled tracer data back to original scale.
        Args:
            scaled_tracer_data: Scaled tracer data
        Returns:
            tracer_original: Tracer data in original scale
        """
        if not hasattr(self, 'tracer_scaler'):
            raise ValueError("Tracer scaler not established. Call establish_scalers() first.")
            
        original_shape = scaled_tracer_data.shape
        tracer_original = self.tracer_scaler.inverse_transform(scaled_tracer_data.reshape(-1, 1)).reshape(original_shape)
        return tracer_original

    def inverse_transform_conductivity_data(self, scaled_conductivity_data):
        """
        Inverse transform scaled conductivity data back to original scale.
        Args:
            scaled_conductivity_data: Scaled conductivity data
        Returns:
            conductivity_original: Conductivity data in original scale
        """
        if not hasattr(self, 'conductivity_scaler'):
            raise ValueError("Conductivity scaler not established. Call establish_scalers() first.")
            
        original_shape = scaled_conductivity_data.shape
        conductivity_original = self.conductivity_scaler.inverse_transform(scaled_conductivity_data.reshape(-1, 1)).reshape(original_shape)
        return conductivity_original

    def get_scaler_info(self):
        """
        Get information about all established scalers for debugging and verification.
        Returns:
            dict: Information about all scalers
        """
        scaler_info = {
            'parameter_scalers': {},
            'ert_scaler': None,
            'tracer_scaler': None,
            'conductivity_scaler': None
        }
        if hasattr(self, 'parameter_scalers') and self.parameter_scalers:
            for i, scaler in self.parameter_scalers.items():
                scaler_info['parameter_scalers'][i] = {
                    'type': type(scaler).__name__,
                    'data_min': scaler.data_min_[0] if hasattr(scaler, 'data_min_') else None,
                    'data_max': scaler.data_max_[0] if hasattr(scaler, 'data_max_') else None,
                    'scale': scaler.scale_[0] if hasattr(scaler, 'scale_') else None
                }
        
        for scaler_name in ['ert_scaler', 'tracer_scaler', 'conductivity_scaler']:
            if hasattr(self, scaler_name):
                scaler = getattr(self, scaler_name)
                scaler_info[scaler_name] = {
                    'type': type(scaler).__name__,
                    'mean': scaler.mean_[0] if hasattr(scaler, 'mean_') else None,
                    'scale': scaler.scale_[0] if hasattr(scaler, 'scale_') else None
                }
        
        return scaler_info

    def well_data_arrange(self, tracer_cont_data, sample_idx=0):
        """
        Arrange tracer concentration data by wells and replicates.
        Args:
            tracer_cont_data: Tracer concentration data array
            sample_idx: Sample index (default=0, set to None for 2D data)
        Returns:
            tracer_arranged: Arranged data with shape (8, 3, 73)
                           - 8 wells, 3 replicates per well, 73 time points
        """
        tracer_arranged = np.zeros((8, 3, 73))
        for w in range(8):
            for rep in range(3):
                if sample_idx is None:
                    tracer_arranged[w, rep, :] = tracer_cont_data[w*3+rep, :]
                else:
                    tracer_arranged[w, rep, :] = tracer_cont_data[sample_idx, w*3+rep, :]
        return tracer_arranged

class ConditionalDataset(Dataset):
    """
    Custom dataset class for loading and processing conditional data.
    """
    def __init__(self, ert_data, tracer_data, conductivity_data, parameter_data):
        self.ert_data = torch.tensor(ert_data, dtype=torch.float32)
        self.tracer_data = torch.tensor(tracer_data, dtype=torch.float32)
        self.parameter_data = torch.tensor(parameter_data, dtype=torch.float32)
        self.conductivity_data = torch.tensor(conductivity_data, dtype=torch.float32)

    def __len__(self):
        return self.ert_data.shape[0]

    def __getitem__(self, idx):
        ert_sample = self.ert_data[idx]
        tracer_sample = self.tracer_data[idx]
        param_sample = self.parameter_data[idx]
        cond_sample = self.conductivity_data[idx]
        return ert_sample, tracer_sample, cond_sample, param_sample
    
# Conditional Diffusion Model
class SpatioTemporalEncoder(nn.Module):
    """
    Encoder module for spatio-temporal data using convolutional layers and transformer layers.
    """
    def __init__(self, spatial_dim, temporal_dim, embedding_dim, num_heads, transformer_layers, dropout=0.1):
        super(SpatioTemporalEncoder, self).__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(in_channels=temporal_dim, out_channels=64, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=64, out_channels=embedding_dim, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout),
            num_layers=transformer_layers
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.spatial_encoder(x)  
        x = x.permute(0, 2, 1)  
        x = self.temporal_encoder(x)  
        return torch.mean(x, dim=1)  

class ConditionalDiffusionModel(nn.Module):
    """Conditional diffusion model for spatio-temporal well observations."""
    def __init__(self, ert_shape, well_obs_shape, param_dim, embedding_dim=128, dropout=0.1):
        super(ConditionalDiffusionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.param_dim = param_dim

        self.ert_encoder = SpatioTemporalEncoder(
            spatial_dim=ert_shape[1], temporal_dim=ert_shape[2],
            embedding_dim=embedding_dim, num_heads=4, transformer_layers=4, dropout=dropout
        )
        self.well_obs_encoder = SpatioTemporalEncoder(
            spatial_dim=well_obs_shape[1], temporal_dim=well_obs_shape[2],
            embedding_dim=embedding_dim, num_heads=4, transformer_layers=4, dropout=dropout
        )
        self.time_embed = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        combined_dim = self.param_dim + 3 * embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, param_dim)
        )
    def forward(self, x, t, ert_data, well_obs_data):
        t_emb = self.get_timestep_embedding(t, self.embedding_dim).to(x.device)
        t_emb = self.time_embed(t_emb)

        ert_embedding = self.ert_encoder(ert_data)
        well_obs_embedding = self.well_obs_encoder(well_obs_data)

        combined_embedding = torch.cat([x, t_emb, ert_embedding, well_obs_embedding], dim=1)
        return self.fc(combined_embedding)

    @staticmethod
    def get_timestep_embedding(timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        exponents = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb_t = timesteps.float().unsqueeze(1) * exponents.unsqueeze(0)
        emb = torch.cat([torch.sin(emb_t), torch.cos(emb_t)], dim=1)
        return emb

class DiffusionScheduler:
    """
    Diffusion scheduler for controlling the noise schedule during training.
    """
    def __init__(self, timesteps, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def get_diffusion_schedule(T, beta_start=1e-4, beta_end=0.02, device='cpu'):
        """Static method to create diffusion schedule without instantiating the class."""
        betas = torch.linspace(beta_start, beta_end, T).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars

    @staticmethod
    def q_sample(x0, t, noise, alpha_bar):
        """Forward diffusion process: add noise to clean data."""

        # Get alpha_bar values for the timesteps and reshape for broadcasting
        alpha_bar_t = alpha_bar[t]  # Shape: [batch_size]
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(-1, 1)  # Shape: [batch_size, 1]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t).view(-1, 1)  # Shape: [batch_size, 1]
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def sample_timesteps(self, batch_size):
        """Sample random timesteps for training."""
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)


# Training Utilities
class TrainingHistoryTracker:
    """
    Class to track and save cumulative training history across all active learning stages.
    """
    def __init__(self, config: Config):
        self.config = config
        self.cumulative_train_losses = []
        self.cumulative_val_losses = []
        self.stage_boundaries = [0]  # Epoch indices where each stage starts
        self.stage_info = []  # Information about each stage
        self.current_epoch = 0
        
    def add_stage_results(self, stage_name, train_losses, val_losses, num_augmented_samples=0):
        """
        Add training results from a stage to the cumulative history.
        
        Args:
            stage_name: Name/identifier for the stage (e.g., "Stage 0", "Stage 1")
            train_losses: Training losses for this stage
            val_losses: Validation losses for this stage
            num_augmented_samples: Number of augmented samples added in this stage
        """
        # Add losses to cumulative history
        self.cumulative_train_losses.extend(train_losses)
        self.cumulative_val_losses.extend(val_losses)
        
        # Update current epoch count
        self.current_epoch += len(train_losses)
        
        # Record stage boundary
        self.stage_boundaries.append(self.current_epoch)
        
        # Store stage information
        stage_data = {
            'name': stage_name,
            'start_epoch': self.stage_boundaries[-2],
            'end_epoch': self.current_epoch,
            'num_epochs': len(train_losses),
            'num_augmented_samples': num_augmented_samples,
            'best_train_loss': min(train_losses),
            'best_val_loss': min(val_losses),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
        self.stage_info.append(stage_data)
        
        print(f"Added {stage_name} to training history:")
        print(f"  Epochs: {stage_data['start_epoch']} - {stage_data['end_epoch']}")
        print(f"  Best Validation Loss: {stage_data['best_val_loss']:.6f}")
        print(f"  Augmented Samples: {num_augmented_samples}")
    
    def save_history(self, filename=None):
        """Save the complete training history to files."""
        if filename is None:
            filename = os.path.join(self.config.RESULTS_DIR, 'cumulative_training_history.npz')
        
        # Save as compressed numpy archive
        np.savez_compressed(
            filename,
            cumulative_train_losses=np.array(self.cumulative_train_losses),
            cumulative_val_losses=np.array(self.cumulative_val_losses),
            stage_boundaries=np.array(self.stage_boundaries),
            stage_info=self.stage_info
        )
        
        # Also save as JSON for readability
        json_filename = filename.replace('.npz', '.json')
        import json
        history_data = {
            'stage_info': self.stage_info,
            'stage_boundaries': self.stage_boundaries,
            'total_epochs': self.current_epoch,
            'num_stages': len(self.stage_info)
        }
        
        with open(json_filename, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Training history saved to:")
        print(f"  Data: {filename}")
        print(f"  Info: {json_filename}")
    
    def plot_cumulative_history(self):
        """Plot the cumulative training history with stage boundaries."""
        if not self.cumulative_train_losses:
            print("No training history to plot.")
            return
        
        epochs = range(len(self.cumulative_train_losses))
        
        plt.figure(figsize=(15, 10))
        
        # Main loss plot
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.cumulative_train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(epochs, self.cumulative_val_losses, label='Validation Loss', color='orange', alpha=0.7)
        
        # Add vertical lines for stage boundaries
        for i, boundary in enumerate(self.stage_boundaries[1:-1], 1):
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.6, 
                       label='Stage Boundary' if i == 1 else '')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Cumulative Training History Across All Active Learning Stages')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale often better for loss visualization
        
        # Stage performance comparison
        plt.subplot(2, 1, 2)
        stage_names = [info['name'] for info in self.stage_info]
        best_val_losses = [info['best_val_loss'] for info in self.stage_info]
        final_val_losses = [info['final_val_loss'] for info in self.stage_info]
        
        x = range(len(stage_names))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], best_val_losses, width, 
                label='Best Validation Loss', alpha=0.8, color='green')
        plt.bar([i + width/2 for i in x], final_val_losses, width, 
                label='Final Validation Loss', alpha=0.8, color='orange')
        
        plt.xlabel('Active Learning Stage')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Comparison Across Stages')
        plt.xticks(x, stage_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, 'cumulative_training_history.png'))
        plt.show(block=False)
        # Print summary statistics
        self.print_summary()
    
    def print_summary(self):
        """Print a summary of training progress across stages."""
        if not self.stage_info:
            print("No stage information available.")
            return
        
        print("\n" + "="*60)
        print("CUMULATIVE TRAINING HISTORY SUMMARY")
        print("="*60)
        print(f"Total Epochs: {self.current_epoch}")
        print(f"Total Stages: {len(self.stage_info)}")
        print()
        
        print(f"{'Stage':<12} {'Epochs':<8} {'Aug. Samples':<12} {'Best Val Loss':<15} {'Improvement':<12}")
        print("-" * 70)
        
        previous_best = None
        for info in self.stage_info:
            improvement = ""
            if previous_best is not None:
                improvement_pct = ((previous_best - info['best_val_loss']) / previous_best) * 100
                improvement = f"{improvement_pct:+.2f}%"
            
            print(f"{info['name']:<12} {info['num_epochs']:<8} {info['num_augmented_samples']:<12} "
                  f"{info['best_val_loss']:<15.6f} {improvement:<12}")
            
            previous_best = info['best_val_loss']
        
        print("\n" + "="*60)
    
    def load_history(self, filename=None):
        """Load previously saved training history."""
        if filename is None:
            filename = os.path.join(self.config.RESULTS_DIR, 'cumulative_training_history.npz')
        
        if not os.path.exists(filename):
            print(f"History file not found: {filename}")
            return False
        
        try:
            data = np.load(filename, allow_pickle=True)
            self.cumulative_train_losses = data['cumulative_train_losses'].tolist()
            self.cumulative_val_losses = data['cumulative_val_losses'].tolist()
            self.stage_boundaries = data['stage_boundaries'].tolist()
            self.stage_info = data['stage_info'].tolist()
            self.current_epoch = len(self.cumulative_train_losses)
            
            print(f"Successfully loaded training history from: {filename}")
            print(f"Total epochs: {self.current_epoch}")
            print(f"Total stages: {len(self.stage_info)}")
            return True
        except Exception as e:
            print(f"Error loading history: {e}")
            return False
    
    @staticmethod
    def load_and_plot_history(config, filename=None):
        """Static method to load and plot training history without creating a new tracker."""
        tracker = TrainingHistoryTracker(config)
        if tracker.load_history(filename):
            tracker.plot_cumulative_history()
            return tracker
        return None

class Trainer:
    """
    Trainer class for managing the training process.
    """
    def __init__(self, model, data_loaders, config: Config, history_tracker=None):
        self.model = model.to(config.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.data_loaders = data_loaders
        self.config = config
        self.history_tracker = history_tracker
        self.diffusion_scheduler = DiffusionScheduler(
            timesteps=config.TDIFFUSION_TIMESTEPS,
            device=config.DEVICE
        )
        
    def train(self, num_epochs=100, checkpoint_path=None, continue_from_checkpoint=False, stage_name=None, num_augmented_samples=0):
        train_losses = []
        val_losses = []
        best_val_loss = np.inf
        start_epoch = 0
        
        # Load checkpoint if continuing training
        if continue_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Continuing training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model state from checkpoint")
            else:
                # Backward compatibility - if only model state is saved
                self.model.load_state_dict(checkpoint)
                print("Loaded model state from checkpoint (legacy format)")
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state from checkpoint")
            
            # Load training progress if available
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"Resuming from epoch {start_epoch}")
            
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                print(f"Previous best validation loss: {best_val_loss:.6f}")
        else:
            print("Starting training from scratch")
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_loss = self._run_epoch(self.data_loaders['train'], training=True)
            val_loss = self._run_epoch(self.data_loaders['val'], training=False)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                print(f"Epoch {epoch + 1}/{start_epoch + num_epochs} - Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Validation improved! Saving BEST model state...")
                self.save_checkpoint(checkpoint_path, epoch, best_val_loss)
        
        # Save last epoch state (for continuing training in next stage)
        if checkpoint_path:
            print(f"Saving LAST EPOCH state for future training continuation...")
            last_checkpoint_path = checkpoint_path.replace('.pt', '_last.pt')
            self.save_checkpoint(last_checkpoint_path, start_epoch + num_epochs - 1, train_losses[-1])
            print(f"Training complete:")
            print(f"  → Best validation model saved to: {checkpoint_path}")
            print(f"  → Last epoch state saved to: {last_checkpoint_path}")
        
        # Add to history tracker if available
        if self.history_tracker and stage_name:
            self.history_tracker.add_stage_results(
                stage_name=stage_name,
                train_losses=train_losses,
                val_losses=val_losses,
                num_augmented_samples=num_augmented_samples
            )
        
        return train_losses, val_losses
    
    def save_checkpoint(self, checkpoint_path, epoch, best_val_loss, best_train_loss=None):
        """Save model, optimizer state, and training progress."""
        if checkpoint_path:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }
            if best_train_loss is not None:
                checkpoint['best_train_loss'] = best_train_loss
            torch.save(checkpoint, checkpoint_path)
    
    def continue_training_from_checkpoint(self, previous_checkpoint_path, new_checkpoint_path, num_epochs, stage_name=None, num_augmented_samples=0):
        """Continue training from a previous checkpoint and save to a new checkpoint.
        Note: This method tracks the best validation loss FOR THIS STAGE ONLY.
        This ensures each stage saves its best model, even if it doesn't beat previous stages.
        """
        train_losses = []
        val_losses = []
        stage_best_val_loss = np.inf  # Track best for THIS stage only
        start_epoch = 0
        
        # Load from previous checkpoint
        if os.path.exists(previous_checkpoint_path):
            print(f"Loading previous checkpoint: {previous_checkpoint_path}")
            checkpoint = torch.load(previous_checkpoint_path)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model state from previous checkpoint")
            else:
                # Backward compatibility
                self.model.load_state_dict(checkpoint)
                print("Loaded model state from previous checkpoint (legacy format)")
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state from previous checkpoint")
            
            # Display previous training info but DON'T use it as baseline
            # We want to save the best model from THIS stage, not compare to previous stages
            if 'best_val_loss' in checkpoint:
                previous_best = checkpoint['best_val_loss']
                print(f"Previous stage best validation loss: {previous_best:.6f}")
                print(f"Note: Will save best model from THIS stage independently")
        else:
            print(f"Previous checkpoint not found: {previous_checkpoint_path}. Starting fresh.")
        
        # Train for additional epochs
        for epoch in range(num_epochs):
            train_loss = self._run_epoch(self.data_loaders['train'], training=True)
            val_loss = self._run_epoch(self.data_loaders['val'], training=False)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")
            
            # Compare only to the best from THIS stage
            if val_loss < stage_best_val_loss:
                stage_best_val_loss = val_loss
                print(f"Stage validation improved! Saving model state for this stage...")
                self.save_checkpoint(new_checkpoint_path, epoch, stage_best_val_loss)
        
        # Add to history tracker if available
        if self.history_tracker and stage_name:
            self.history_tracker.add_stage_results(
                stage_name=stage_name,
                train_losses=train_losses,
                val_losses=val_losses,
                num_augmented_samples=num_augmented_samples
            )
        
        return train_losses, val_losses
    
    def continue_training_from_last_epoch(self, previous_checkpoint_path, new_checkpoint_path, num_epochs, stage_name=None, num_augmented_samples=0):
        """Continue training from the LAST EPOCH (not best validation) of previous checkpoint.
        
        This method loads the last training state and continues from there, while tracking
        both training and validation losses to save the best model based on either metric.
        
        Saves three checkpoints:
        - new_checkpoint_path.replace('.pt', '_best_val.pt'): Best validation model
        - new_checkpoint_path.replace('.pt', '_best_train.pt'): Best training model  
        - new_checkpoint_path.replace('.pt', '_last.pt'): Last epoch state (for continued training)
        
        Args:
            previous_checkpoint_path: Path to previous stage checkpoint
            new_checkpoint_path: Path to save best model for this stage
            num_epochs: Number of epochs to train
            stage_name: Name of current stage for tracking
            num_augmented_samples: Number of augmented samples added
            
        Returns:
            tuple: (train_losses, val_losses) for this stage
        """
        train_losses = []
        val_losses = []
        stage_best_val_loss = np.inf  # Track best validation loss for THIS stage
        stage_best_train_loss = np.inf  # Track best training loss for THIS stage
        start_epoch = 0
        
        # Determine which checkpoint to load from (last epoch or best if last doesn't exist)
        previous_last_checkpoint = previous_checkpoint_path.replace('.pt', '_last.pt')
        load_from = previous_last_checkpoint if os.path.exists(previous_last_checkpoint) else previous_checkpoint_path
        
        # Load from previous checkpoint (prefer last epoch state)
        if os.path.exists(load_from):
            print(f"Loading checkpoint: {load_from}")
            if load_from == previous_last_checkpoint:
                print("  → Loading from LAST EPOCH state (continuing from last training point)")
            else:
                print("  → Loading from BEST MODEL state (last epoch checkpoint not found)")
                
            checkpoint = torch.load(load_from)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model state from checkpoint")
            else:
                # Backward compatibility
                self.model.load_state_dict(checkpoint)
                print("Loaded model state from checkpoint (legacy format)")
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state from checkpoint")
            
            # Display previous training info
            if 'epoch' in checkpoint:
                print(f"Previous training epoch: {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                previous_best = checkpoint['best_val_loss']
                print(f"Previous best validation loss: {previous_best:.6f}")
                print(f"Note: Will track and save best model from THIS stage independently")
        else:
            print(f"Checkpoint not found: {load_from}. Starting fresh.")
        
        # Train for additional epochs
        for epoch in range(num_epochs):
            train_loss = self._run_epoch(self.data_loaders['train'], training=True)
            val_loss = self._run_epoch(self.data_loaders['val'], training=False)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")
            
            # Save best validation model
            if val_loss < stage_best_val_loss:
                stage_best_val_loss = val_loss
                print(f"  → Validation improved! Saving best validation model...")
                best_val_checkpoint_path = new_checkpoint_path.replace('.pt', '_best_val.pt')
                self.save_checkpoint(best_val_checkpoint_path, epoch, stage_best_val_loss, stage_best_train_loss)
            
            # Save best training model
            if train_loss < stage_best_train_loss:
                stage_best_train_loss = train_loss
                print(f"  → Training improved! Saving best training model...")
                best_train_checkpoint_path = new_checkpoint_path.replace('.pt', '_best_train.pt')
                self.save_checkpoint(best_train_checkpoint_path, epoch, stage_best_val_loss, stage_best_train_loss)
        
        print(f"\nSaving LAST EPOCH state for future training continuation...")
        last_checkpoint_path = new_checkpoint_path.replace('.pt', '_last.pt')
        self.save_checkpoint(last_checkpoint_path, num_epochs - 1, val_losses[-1], train_losses[-1])
        
        # Add to history tracker if available
        if self.history_tracker and stage_name:
            self.history_tracker.add_stage_results(
                stage_name=stage_name,
                train_losses=train_losses,
                val_losses=val_losses,
                num_augmented_samples=num_augmented_samples
            )
        
        # Determine which model is best for inference
        best_val_checkpoint_path = new_checkpoint_path.replace('.pt', '_best_val.pt')
        best_train_checkpoint_path = new_checkpoint_path.replace('.pt', '_best_train.pt')
        
        # Check which checkpoint files exist
        best_val_exists = os.path.exists(best_val_checkpoint_path)
        best_train_exists = os.path.exists(best_train_checkpoint_path)
        
        # Select the best model based on lower loss (with fallback to last checkpoint)
        if not best_val_exists and not best_train_exists:
            # No improvement occurred during training - use last epoch
            print(f"\n  → No improvement during training (validation loss: {stage_best_val_loss:.6f}, train loss: {stage_best_train_loss:.6f})")
            print(f"  → Using LAST EPOCH model for inference in next stage")
            best_inference_checkpoint = last_checkpoint_path
        elif best_val_exists and (not best_train_exists or stage_best_val_loss < stage_best_train_loss):
            print(f"\n  → Best validation loss ({stage_best_val_loss:.6f}) < Best training loss ({stage_best_train_loss:.6f})")
            print(f"  → Using VALIDATION model for inference in next stage")
            best_inference_checkpoint = best_val_checkpoint_path
        else:
            print(f"\n  → Best training loss ({stage_best_train_loss:.6f}) <= Best validation loss ({stage_best_val_loss:.6f})")
            print(f"  → Using TRAINING model for inference in next stage")
            best_inference_checkpoint = best_train_checkpoint_path
        
        # Copy best inference model to main checkpoint path for next stage
        import shutil
        shutil.copy(best_inference_checkpoint, new_checkpoint_path)
        
        print(f"\nStage {stage_name} complete:")
        print(f"  → Best validation model saved to: {best_val_checkpoint_path}")
        print(f"  → Best training model saved to: {best_train_checkpoint_path}")
        print(f"  → Best inference model (for next stage): {new_checkpoint_path}")
        print(f"  → Last epoch state saved to: {last_checkpoint_path}")
        
        return train_losses, val_losses
    
    def load_model_only(self, checkpoint_path):
        """Load only the model state (for inference, not continuing training)."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Backward compatibility
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from: {checkpoint_path}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

    def _run_epoch(self, loader, training=True):
        epoch_loss = 0.0
        self.model.train(training)
        
        for ert_data, tracer_data, conductivity_data, param_data in loader:
            ert_data = ert_data.to(self.config.DEVICE)
            well_obs_data = conductivity_data.to(self.config.DEVICE)
            param_data = param_data.to(self.config.DEVICE)

            B = param_data.size(0)  
            t = self.diffusion_scheduler.sample_timesteps(B)
            noise = torch.randn_like(param_data)
            x_noisy = DiffusionScheduler.q_sample(param_data, t, noise, self.diffusion_scheduler.alpha_bars)
            pred_noise = self.model(x_noisy, t, ert_data, well_obs_data)
            loss = self.criterion(pred_noise, noise)
            epoch_loss += loss.item() * B

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return epoch_loss / len(loader.dataset)

class ActiveLearningLoop:
    """
    Active learning loop for iterative model improvement.
    """
    def __init__(self, trainer, config: Config, pflotran_module, data_handler: DataLoaderHandler):
        self.trainer = trainer
        self.config = config
        self.pflotran_module = pflotran_module  
        self.data_handler = data_handler 
        self.diffusion_scheduler = DiffusionScheduler(
            timesteps=config.TDIFFUSION_TIMESTEPS, 
            device=config.DEVICE
        )
        # Running archive of (normalized) parameters and their WSSE for guided sampling
        self.wsse_archive_params = []
        self.wsse_archive_scores = []
        self.field_fit_history = []
        self.field_fit_log = os.path.join(self.config.RESULTS_DIR, 'field_fit_history.json')
        self.cdm_eval_history = []
        self.cdm_eval_log = os.path.join(self.config.RESULTS_DIR, 'cdm_field_eval_history.json')
        if os.path.exists(self.field_fit_log):
            try:
                with open(self.field_fit_log, 'r') as f:
                    self.field_fit_history = json.load(f)
            except Exception as exc:
                print(f"[WARN] Could not load existing field fit history: {exc}")
        if os.path.exists(self.cdm_eval_log):
            try:
                with open(self.cdm_eval_log, 'r') as f:
                    self.cdm_eval_history = json.load(f)
            except Exception as exc:
                print(f"[WARN] Could not load existing CDM evaluation history: {exc}")
    
    def wsse_metric(self, predictions, observations):
        """
        Calculate Weighted Sum of Squared Errors (WSSE) metric.
        Args:
            A: Coefficient for relative error scaling
            B: Coefficient for absolute error scaling  
            predictions: Predicted values
            observations: Observed values
        Returns:
            tuple: (WSSE, WSE) - Overall WSSE and individual WSE values
        """
        sd = self.config.ALPHA_SD * np.abs(observations) + self.config.BETA_SD
        WSE = (predictions - observations)**2 / (sd)**2
        WSSE = np.average(WSE)
        return WSSE, WSE

    def evaluate_simulations_against_field(self, cdm_ertd, field_ert_true):
        """
        Evaluate CDM simulations against field data using WSSE metric.
        Args:
            cdm_ertd: CDM ERT data simulations
            field_ert_true: Field ERT data (true values)   
        Returns:
            WSSE_sim, wsse_total, sorted_indices - Time-series WSSE, total WSSE, sorted indices
        """
        print("Evaluating CDM simulations against field data...")
        WSSE_sim = []
        for sim in range(cdm_ertd.shape[0]):
            WSSE_time = []
            for es in range(cdm_ertd.shape[2]):
                WSSE, _ = self.wsse_metric(cdm_ertd[sim][:, es], field_ert_true[:, es])
                WSSE_time.append(WSSE)
            WSSE_sim.append(WSSE_time)
        WSSE_sim = np.array(WSSE_sim)  

        wsse_total = WSSE_sim.sum(axis=1)
        sorted_indices = np.argsort(wsse_total)

        wsse_global = []
        for i in range(len(cdm_ertd)):
            wsse, _ = self.wsse_metric(cdm_ertd[i], field_ert_true)
            wsse_global.append(wsse)
        wsse_global = np.array(wsse_global)

        return WSSE_sim, wsse_global, sorted_indices

    def update_wsse_archive(self, params_norm, wsse_scores):
        """Persist normalized parameters and their WSSE scores for future sampling."""
        if params_norm is None or wsse_scores is None:
            return
        params_norm = np.asarray(params_norm)
        wsse_scores = np.asarray(wsse_scores).reshape(-1)
        if params_norm.size == 0 or wsse_scores.size == 0:
            return
        if params_norm.ndim == 1:
            params_norm = params_norm[None, :]
        if params_norm.shape[0] != wsse_scores.shape[0]:
            raise ValueError("Parameter array and WSSE scores must share the same length")
        self.wsse_archive_params.extend(params_norm.tolist())
        self.wsse_archive_scores.extend(wsse_scores.tolist())

    def _archive_ready(self, min_samples=10):
        return len(self.wsse_archive_params) >= min_samples

    def record_field_fit_metrics(self, stage, wsse_total, sorted_indices, params_norm):
        if wsse_total.size == 0 or params_norm.size == 0:
            return
        if params_norm.ndim == 1:
            params_norm = params_norm[None, :]
        best_idx = int(sorted_indices[0]) if len(sorted_indices) > 0 else int(np.argmin(wsse_total))
        best_wsse = float(wsse_total[best_idx])
        percentiles = {
            'p05': float(np.percentile(wsse_total, 5)),
            'p25': float(np.percentile(wsse_total, 25)),
            'p50': float(np.percentile(wsse_total, 50)),
            'p75': float(np.percentile(wsse_total, 75)),
            'p95': float(np.percentile(wsse_total, 95)),
        }
        best_params_norm = params_norm[best_idx]
        best_params_orig = self.data_handler.inverse_transform_parameters(best_params_norm.reshape(1, -1))[0].tolist()
        entry = {
            'stage': stage,
            'best_wsse': best_wsse,
            'mean_wsse': float(np.mean(wsse_total)),
            'min_wsse': float(wsse_total.min()),
            'max_wsse': float(wsse_total.max()),
            'percentiles': percentiles,
            'best_params_norm': best_params_norm.tolist(),
            'best_params_original': best_params_orig,
        }
        self.field_fit_history.append(entry)
        try:
            with open(self.field_fit_log, 'w') as f:
                json.dump(self.field_fit_history, f, indent=2)
        except Exception as exc:
            print(f"[WARN] Failed to write field fit history: {exc}")

    def evaluate_cdm_field_fit(self, stage, field_ert_tensor, field_well_tensor, field_ert_true):
        num_eval_samples = min(self.config.FIELD_EVAL_SAMPLE_COUNT, self.config.NUM_SAMPLES)
        if num_eval_samples <= 0:
            return
        sampled_params = self.sample_model_with_clipping(
            ert_data=field_ert_tensor,
            well_obs_data=field_well_tensor,
            params_dim=self.trainer.model.param_dim,
            num_samples=num_eval_samples,
            conditioning_noise_std=0.0,
            use_diverse_conditioning=False
        )
        if sampled_params is None or len(sampled_params) == 0:
            print(f"[WARN] CDM field evaluation skipped at stage {stage}: no samples generated")
            return
        
        # Generate CDM ERT data simulations in Paralell 
        cdm_ertd = pflotran_simulate_parallel(
            pflotran_module=self.pflotran_module,
            data_handler=self.data_handler,
            cdm_parameters=sampled_params,
            config=self.config,
            batch_size=self.config.FIELD_EVAL_BATCH_SIZE,
            verbose=False)

        if cdm_ertd is None or cdm_ertd.size == 0:
            print(f"[WARN] CDM field evaluation skipped at stage {stage}: no simulation results")
            return


        entry = {
            'stage': stage,
            'best_wsse': float(np.min(wsse_values)),
            'mean_wsse': float(np.mean(wsse_values)),
            'median_wsse': float(np.median(wsse_values)),
            'num_samples': len(wsse_values),
        }
        self.cdm_eval_history.append(entry)
        try:
            with open(self.cdm_eval_log, 'w') as f:
                json.dump(self.cdm_eval_history, f, indent=2)
        except Exception as exc:
            print(f"[WARN] Failed to write CDM evaluation history: {exc}")

        print(f"[CDM FIELD EVAL] Stage {stage}: best WSSE={entry['best_wsse']:.4f}, "
              f"median={entry['median_wsse']:.4f}, mean={entry['mean_wsse']:.4f}")

    def wsse_guided_sampling(
        self,
        num_samples,
        exploit_frac=0.6,
        noise_floor=0.01,
        noise_ceiling=0.2,
        min_archive=20,
    ):
        """Sample new parameters by perturbing archived low-WSSE points while keeping diversity."""
        if not self._archive_ready(min_archive):
            return None

        params = np.array(self.wsse_archive_params, dtype=np.float32)
        scores = np.array(self.wsse_archive_scores, dtype=np.float32)
        sort_idx = np.argsort(scores)
        params = params[sort_idx]
        scores = scores[sort_idx]

        dim = params.shape[1]
        if params.shape[0] == 1:
            cov = np.eye(dim, dtype=np.float32) * 1e-3
        else:
            cov = np.cov(params, rowvar=False)
            if cov.ndim == 0:
                cov = np.eye(dim, dtype=np.float32) * 1e-3
        cov += np.eye(dim, dtype=np.float32) * 1e-6

        exploit_total = max(1, int(num_samples * exploit_frac))
        explore_total = max(0, num_samples - exploit_total)
        exploit_pool = params[: max(exploit_total * 2, 1)]
        exploit_scores = scores[: exploit_pool.shape[0]]
        rng = np.random.default_rng()
        proposals = []
        median_wsse = np.median(scores)
        median_wsse = median_wsse if median_wsse > 0 else 1.0

        def perturb(seed, seed_score, scale_multiplier=1.0):
            relative = np.clip(seed_score / median_wsse, 0.25, 4.0)
            noise_scale = np.clip(relative * scale_multiplier, noise_floor, noise_ceiling)
            noise = rng.multivariate_normal(np.zeros(dim), cov)
            proposal = np.clip(seed + noise * noise_scale, 0.0, 1.0)
            return proposal

        for _ in range(exploit_total):
            idx = rng.integers(0, exploit_pool.shape[0])
            proposals.append(perturb(exploit_pool[idx], exploit_scores[idx], scale_multiplier=0.5))

        if explore_total > 0:
            explore_pool = params[max(exploit_total * 2, 1):]
            explore_scores = scores[max(exploit_total * 2, 1):]
            if explore_pool.size == 0:
                explore_pool = params
                explore_scores = scores
            for _ in range(explore_total):
                idx = rng.integers(0, explore_pool.shape[0])
                proposals.append(perturb(explore_pool[idx], explore_scores[idx], scale_multiplier=1.5))

        proposals = np.array(proposals, dtype=np.float32)
        proposals = np.clip(proposals, 0.0, 1.0)
        proposals = proposals[:num_samples]
        return self.data_handler.inverse_transform_parameters(proposals)

    def Uncertainty_sampling(self, cdm_parameters_norm, wsse_total, top_k=10, sample_repeats=5):
        """
        Generate new parameter samples based on uncertainty and misfit analysis.
        Args:
            cdm_parameters_norm: Normalized CDM parameters
            wsse_total: Total WSSE scores
            top_k: Number of top samples to select
            sample_repeats: Number of new samples per top sample
        Returns:
            tuple: (augmented_samples_norm, augmented_samples) - Normalized and original scale samples
        """
        global_means = np.mean(cdm_parameters_norm, axis=0)  
        global_std_devs = np.std(cdm_parameters_norm, axis=0)  
        z_scores = (cdm_parameters_norm - global_means) / global_std_devs  

        total_uncertainty_scores = np.mean(np.abs(z_scores), axis=1)

        wsse_total_norm = wsse_total / wsse_total.max()
        uncertainty_norm = total_uncertainty_scores / total_uncertainty_scores.max()
        
        alpha = 0.1  # Weighting factor between 0 and 1
        q_score = alpha * wsse_total_norm + (1 - alpha) * uncertainty_norm

        ranked_q_indices = np.argsort(-q_score)
        top_k_indices = ranked_q_indices[:top_k]
        top_k_samples = cdm_parameters_norm[top_k_indices]
        attempt = 0

        augmented_samples_norm = []
        for i in range(top_k_samples.shape[0]):
            generated = 0
            while generated < sample_repeats and attempt < self.config.MAX_AUGMENTED_SAMPLE_ATTEMPTS:
                attempt += 1
                sample = top_k_samples[i]
                
                # Use scaled Gaussian noise centered at 0 (instead of uniform 0-1 noise)
                # Scale by uncertainty and WSSE to focus perturbations on uncertain/high-error regions
                noise_scale = 0.1 * uncertainty_norm[top_k_indices[i]] * wsse_total_norm[top_k_indices[i]]
                noise = np.random.randn(sample.shape[0]) * noise_scale  # Gaussian noise
                
                # Add noise and clip to valid range [0, 1]
                new_sample = np.clip(sample + noise, 0.0, 1.0)
                
                # Check if sample is different enough from original (avoid duplicates)
                if np.linalg.norm(new_sample - sample) > 1e-6:
                    augmented_samples_norm.append(new_sample)
                    generated += 1
        if attempt >= self.config.MAX_AUGMENTED_SAMPLE_ATTEMPTS:
            print(f"[WARNING] Max attempts reached for augmented sampling. Generated {len(augmented_samples_norm)} samples.")
        
        augmented_samples_norm = np.array(augmented_samples_norm)
        
        if augmented_samples_norm.shape[0] == 0:
            print(f"[ERROR] No valid samples generated! Returning empty arrays.")
            return np.array([]), np.array([])
        
        augmented_samples = self.data_handler.inverse_transform_parameters(augmented_samples_norm)
        print(f"Successfully generated {augmented_samples.shape[0]} augmented samples")
        print(f"  Augmented samples shape: {augmented_samples.shape}")

        return augmented_samples_norm, augmented_samples

    def GP_sampling(self,cdm_parameters_norm, wsse_total, num_samples=50):
        """
        Generate new parameter samples using Gaussian Process regression.
        Args:
            cdm_parameters_norm: Normalized CDM parameters (can be cumulative across stages)
            wsse_total: Total WSSE scores (can be cumulative across stages)
            num_samples: Number of new samples to generate
        Returns:
            tuple: (gp_samples_norm, gp_samples) - Normalized and original scale samples
        """
        
        wsse_global_norm = (wsse_total - wsse_total.min()) / (wsse_total.max() - wsse_total.min())
        
        # Fit Gaussian Process model
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 10))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-5, normalize_y=True)  # GP model
        gp.fit(cdm_parameters_norm, wsse_global_norm)
        
        
        #Generate new samples and predict expected misfit from CDM samples
        new_samples = np.random.rand(num_samples*2, cdm_parameters_norm.shape[1]) #Larger pool to select from
        predicted_mean, predicted_std = gp.predict(new_samples, return_std=True)

        # Select top-k samples based on lowest predicted mean
        exploitation_indices = np.argsort(predicted_mean)[:num_samples]  # Samples with lowest predicted misfit
        prioritized_samples_norm = new_samples[exploitation_indices]  # Select top-k samples
        
        #Check valid range [0, 1]
        prioritized_samples_norm = self.screen_parameters(prioritized_samples_norm, lower_bound=0.0, upper_bound=1.0)
        augmented_samples = self.data_handler.inverse_transform_parameters(prioritized_samples_norm)
        
        return prioritized_samples_norm, augmented_samples

    def sample_model(self, model, ert_data, well_obs_data, params_dim, temperature=1.0, num_steps=None):
        """
        Reverse sample from the trained CDM model:
        Generate parameter realizations based on conditional ERT and tracer data.
        """
        model.eval()
        with torch.no_grad():
            # Ensure input data is on the same device as the model
            ert_data = ert_data.to(self.config.DEVICE)
            well_obs_data = well_obs_data.to(self.config.DEVICE)
            
            B = ert_data.size(0)  # Batch size
            x = torch.randn(B, params_dim, device=self.config.DEVICE)
            
            # Timesteps for diffusion (reversed)
            timesteps = self.config.TDIFFUSION_TIMESTEPS
            if num_steps is None:
                num_steps = timesteps

            betas, alphas, alpha_bar = DiffusionScheduler.get_diffusion_schedule(
                T=timesteps, 
                beta_start=1e-4, 
                beta_end=0.02, 
                device=self.config.DEVICE
            )
            # Perform reverse diffusion
            for t in reversed(range(num_steps)):
                t_tensor = torch.full((B,), t, device=self.config.DEVICE, dtype=torch.long)
                pred_noise = model(x, t_tensor, ert_data, well_obs_data)
                alpha_t = alphas[t]
                alpha_bar_t = alpha_bar[t]
                coef = (1 - alpha_t) / (math.sqrt(1 - alpha_bar_t) + 1e-8)
                x = (1.0 / math.sqrt(alpha_t)) * (x - coef * pred_noise)

                # Add Gaussian noise for sampling variability
                if t > 0:
                    z = torch.randn_like(x)
                    sigma_t = math.sqrt(betas[t])
                    x = x + sigma_t * temperature * z

            return x  # Return sampled parameters

    def adaptive_sample_model(self, ert_data, well_obs_data, params_dim, 
                             num_samples=50, initial_temperature=1.0,
                             max_temperature=1.0, min_temperature=0.2, temp_decay=0.9,
                             temp_increase=1.02, max_invalid_attempts_to_decay=10,
                             conditioning_noise_std=0.3, use_diverse_conditioning=True):
        """
        Adaptive sampling with temperature adjustment and DIVERSE CONDITIONING to prevent mode collapse.
        
        CRITICAL FIX: Adds noise to conditioning data to generate diverse parameter samples.
        Without this, all samples converge to a single solution because the model always
        receives the same field data as input.
        
        Args:
            ert_data: ERT field data (reference)
            well_obs_data: Well observation data (tracer or conductivity)
            params_dim: Parameter dimension
            num_samples: Number of valid samples to collect
            initial_temperature: Starting temperature
            max_temperature: Upper bound for temperature
            min_temperature: Lower bound for temperature
            temp_decay: Factor to reduce temperature when invalid samples persist
            temp_increase: Factor to increase temperature when valid samples are found
            max_invalid_attempts_to_decay: Trigger adaptive adjustment after consecutive invalid samples
            conditioning_noise_std: Std deviation of noise added to conditioning (higher = more diversity)
            use_diverse_conditioning: If True, adds noise to conditioning; if False, uses field data directly
            
        Returns:
            np.array: Valid parameter samples
            list: Temperature history for debugging
        """        
        temperature = initial_temperature
        cdm_parameters_norm = [] 
        attempt = 0  
        invalid_attempts = 0  
        temperature_history = []  

        while len(cdm_parameters_norm) < num_samples and attempt < self.config.MAX_AUGMENTED_SAMPLE_ATTEMPTS:
            attempt += 1
            temperature_history.append(temperature)
            
            # CRITICAL FIX: Add noise to conditioning to create diverse samples
            if use_diverse_conditioning and conditioning_noise_std > 0:
                # Add Gaussian noise to conditioning channels for diversity
                noise_ert = torch.randn_like(ert_data) * conditioning_noise_std
                noise_well = torch.randn_like(well_obs_data) * conditioning_noise_std
                
                cond_ert = ert_data + noise_ert
                cond_well_obs = well_obs_data + noise_well
            else:
                cond_ert = ert_data
                cond_well_obs = well_obs_data
  
            cdm_parameters = self.sample_model(
                model=self.trainer.model,
                ert_data=cond_ert, 
                well_obs_data=cond_well_obs,
                params_dim=params_dim,
                temperature=temperature
            )           

            valid_cdm_parameters = self.screen_parameters(
                cdm_parameters.detach().cpu().numpy(), 
                lower_bound=0.0, 
                upper_bound=1.0
            )
            if valid_cdm_parameters.shape[0] == 0:
                invalid_attempts += 1
                # Reduce temperature if invalid attempts exceed threshold
                if invalid_attempts >= max_invalid_attempts_to_decay:
                    temperature = max(temperature * temp_decay, min_temperature)
                    invalid_attempts = 0 
            else:
                cdm_parameters_norm.extend(valid_cdm_parameters)
                invalid_attempts = 0 
                # Gradually restore temperature if valid samples found
                temperature = min(temperature * temp_increase, max_temperature)            
            print(f'  Collected {len(cdm_parameters_norm)}/{num_samples} samples.')

        cdm_parameters_norm = np.array(cdm_parameters_norm)
        # Inverse transform normalized parameters back to original scale
        cdm_parameters_orig = self.data_handler.inverse_transform_parameters(cdm_parameters_norm)
        return cdm_parameters_orig, temperature_history

    def screen_parameters(self, sampled_params, lower_bound=0.0, upper_bound=1.0):
        """
        Screen samples by ensuring all parameters fall within [0, 1] range.
        """
        valid_mask = ((sampled_params >= lower_bound) & (sampled_params <= upper_bound)).all(axis=1)
        return sampled_params[valid_mask]
    
    def sample_model_with_clipping(self, ert_data, well_obs_data, params_dim, 
                                   num_samples=50, temperature=1.0,
                                   conditioning_noise_std=0.0, use_diverse_conditioning=False,
                                   lower_bound=0.0, upper_bound=1.0):
        """
        Sample from the CDM model and clip parameters to [lower_bound, upper_bound] range.
        
        Unlike adaptive_sample_model, this function does NOT reject samples - it clips them
        to the valid range. This is more efficient as it generates exactly num_samples without
        rejection/retry logic.
        
        Args:
            ert_data: ERT field data (reference)
            well_obs_data: Well observation data (tracer or conductivity)
            params_dim: Parameter dimension
            num_samples: Number of samples to generate
            temperature: Sampling temperature for diffusion process
            conditioning_noise_std: Std deviation of noise added to conditioning
            use_diverse_conditioning: If True, adds noise to conditioning data
            lower_bound: Lower bound for clipping (default: 0.0)
            upper_bound: Upper bound for clipping (default: 1.0)
            
        Returns:
            np.array: Clipped parameter samples in original scale
        """
        self.trainer.model.eval()
        
        print(f"\n[CLIPPING SAMPLER] Generating {num_samples} samples with parameter clipping...")
        print(f"  Temperature: {temperature}")
        print(f"  Conditioning noise std: {conditioning_noise_std}")
        print(f"  Clipping range: [{lower_bound}, {upper_bound}]")
        
        # Expand conditioning data to match num_samples (repeat for batch processing)
        # ert_data and well_obs_data typically have shape [1, ...] for single field observation
        ert_data_expanded = ert_data.repeat(num_samples, 1, 1) if ert_data.dim() == 3 else ert_data.repeat(num_samples, 1)
        well_obs_expanded = well_obs_data.repeat(num_samples, 1, 1) if well_obs_data.dim() == 3 else well_obs_data.repeat(num_samples, 1)
        
        # Add noise to conditioning if requested (for diversity)
        if use_diverse_conditioning and conditioning_noise_std > 0:
            noise_ert = torch.randn_like(ert_data_expanded) * conditioning_noise_std
            noise_well = torch.randn_like(well_obs_expanded) * conditioning_noise_std
            cond_ert = ert_data_expanded + noise_ert
            cond_well_obs = well_obs_expanded + noise_well
        else:
            cond_ert = ert_data_expanded
            cond_well_obs = well_obs_expanded
        
        # Sample from CDM (now with batch size = num_samples)
        cdm_parameters = self.sample_model(
            model=self.trainer.model,
            ert_data=cond_ert,
            well_obs_data=cond_well_obs,
            params_dim=params_dim,
            temperature=temperature
        )
        
        # Convert to numpy
        cdm_parameters_norm = cdm_parameters.detach().cpu().numpy()
        
        # Clip parameters to valid range
        cdm_parameters_clipped = np.clip(cdm_parameters_norm, lower_bound, upper_bound)
        
        # Report clipping statistics
        num_clipped = np.sum((cdm_parameters_norm < lower_bound) | (cdm_parameters_norm > upper_bound))
        total_params = cdm_parameters_norm.size
        clip_percentage = (num_clipped / total_params) * 100
        
        print(f"  Clipped {num_clipped}/{total_params} parameter values ({clip_percentage:.2f}%)")
        print(f"  Generated {cdm_parameters_clipped.shape[0]} samples (all valid)")
        
        # Inverse transform to original scale
        cdm_parameters_orig = self.data_handler.inverse_transform_parameters(cdm_parameters_clipped)
        
        return cdm_parameters_orig
    
    def plot_temperature_history(self, temperature_history, title="Temperature Adaptation During Sampling", save_path=None):
        """
        Plot the temperature adjustment history during adaptive sampling.
        Args:
            temperature_history: List of temperature values over sampling attempts
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(temperature_history, 'b-', linewidth=2, marker='o', markersize=4)
        plt.title(title, fontsize=14)
        plt.xlabel('Sampling Attempt', fontsize=12)
        plt.ylabel('Temperature', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        plt.show(block=False)
        
        # Print statistics
        print(f"Temperature Statistics:")
        print(f"  Initial: {temperature_history[0]:.4f}")
        print(f"  Final: {temperature_history[-1]:.4f}")
        print(f"  Min: {min(temperature_history):.4f}")
        print(f"  Max: {max(temperature_history):.4f}")
        print(f"  Mean: {np.mean(temperature_history):.4f}")
    

    def pflotran_simulate_parallel(self, param_array, learning_stage=None, case_name='test'):
        """
        Run PFLOTRAN simulations in parallel and save results.
        
        Args:
            param_array: Array of parameters to simulate
            learning_stage: Learning stage number (None for test case)
            case_name: Case identifier ('Test', 'WSSE_guided', or 'Active_CDM')
        """
        results_dir, sim_out_dir = self.config.LEARNING_STAGE_DIR, self.config.SIM_OUT_DIR
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(sim_out_dir):
            os.makedirs(sim_out_dir)

        print(f"Running PFLOTRAN simulations for {param_array.shape[0]} samples.")
        
        # Determine simulation naming and file suffix based on case
        if learning_stage is None:
            # Test case: field parameter verification
            sim_name = "test_data"
            print(f"  Case: {case_name} (field parameter verification)")
        else:
            # Active learning stage
            sim_name = f"{case_name}_{learning_stage}_"
            print(f"  Case: {case_name} at learning stage {learning_stage}")
        
        pf.run_sim_par(self.config.PFTEMPLATE, param_array.T, sim_name, use_limits=True, delete_files=True)
        
        print(f"\n=== POST-PFLOTRAN FILE CHECK ===")
        import glob
        
        # Check all .npy files in current directory
        all_npy = glob.glob("*.npy")
        print(f"Total .npy files in current dir: {len(all_npy)}")
        
        # Check specifically for simulation files
        sim_files = glob.glob(f"*{sim_name}*.npy")
        print(f"Files matching '{sim_name}': {len(sim_files)}")
        if len(sim_files) > 0:
            print(f"First 5 matching files:")
            for f in sorted(sim_files)[:5]:
                print(f"  {f}")
        
        # Check what the expected pattern would be
        expected_pattern = self.config.PFTEMPLATE.split('.in')[0] + f"_{sim_name}"
        expected_files = glob.glob(f"{expected_pattern}*.npy")
        print(f"Files matching expected pattern '{expected_pattern}*': {len(expected_files)}")
        if len(expected_files) > 0:
            print(f"First 5 expected files:")
            for f in sorted(expected_files)[:5]:
                print(f"  {f}")
        print(f"=== END FILE CHECK ===\n")

        # Rename and move simulation files
        print(f"Renaming and moving simulation output files...")
        
        moved_count = 0
        missing_count = 0
        
        for i, param_sample in enumerate(param_array):
            snum = '{0:-05d}'.format(i)
            pre = self.config.PFTEMPLATE.split('.in')[0] + "_" + sim_name + "_" + snum + "_"
            
            # Determine output file naming based on case
            if learning_stage is None:
                # Test case: {tag}_{i}_Test.npy
                post_suffix = f"_{case_name}"
                if i < 3:
                    print(f"\nProcessing {case_name} simulation {i}:")
                for tag in ['obst', 'obsd', 'ertt', 'ertd']:
                    fin = pre + f'{tag}.npy'
                    fout = os.path.join(results_dir, f'{tag}_{i}{post_suffix}.npy')
                    if os.path.exists(fin):
                        os.rename(fin, fout)
                        moved_count += 1
                        if i < 3:  # Only print details for first 3
                            print(f"  ✓ Moved: {fin} -> {os.path.basename(fout)}")
                    else:
                        missing_count += 1
                        if i < 3:  # Only print warnings for first 3
                            print(f"  ✗ NOT FOUND: {fin}")
                np.save(os.path.join(results_dir, f"params_{i}{post_suffix}.npy"), param_array[i])
            else:
                # Active learning stage: {tag}_{case_name}_{stage}_{i}_stage_{stage}.npy
                post_suffix = f"_stage_{learning_stage}"
                if i < 3:
                    print(f"\nProcessing {case_name} simulation {i} (stage {learning_stage}):")
                for tag in ['obst', 'obsd', 'ertt', 'ertd']:
                    fin = pre + f'{tag}.npy'
                    fout = os.path.join(results_dir, f'{tag}_{case_name}_{learning_stage}_{i}{post_suffix}.npy')
                    if os.path.exists(fin):
                        os.rename(fin, fout)
                        moved_count += 1
                        if i < 3:  # Only print details for first 3
                            print(f"  ✓ Moved: {fin} -> {os.path.basename(fout)}")
                    else:
                        missing_count += 1
                        if i < 3:  # Only print warnings for first 3
                            print(f"  ✗ NOT FOUND: {fin}")
                np.save(os.path.join(results_dir, f"params_{case_name}_{learning_stage}_{i}{post_suffix}.npy"), param_array[i])
        
        print(f"\nSummary: Moved {moved_count} files, {missing_count} files missing")
        
        if learning_stage is None:
            print(f"PFLOTRAN {case_name} simulations completed (verification run).")
        else:
            print(f"PFLOTRAN {case_name} simulations completed for stage {learning_stage}.")

    def pflotran_simulate(self, param_array, learning_stage):
        """
        Run PFLOTRAN simulations for the generated samples and save their outputs.
        """
        results_dir, sim_out_dir = self.config.LEARNING_STAGE_DIR, self.config.SIM_OUT_DIR
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(sim_out_dir):
            os.makedirs(sim_out_dir)

        print(f"Running PFLOTRAN simulations for {param_array.shape[0]} samples.")
        for i, param_sample in enumerate(param_array):
            sim_name = f"augmented_{learning_stage}_sim_{i}"
            ertt, ertd, obst, obsd = self.pflotran_module.run_sim(self.config.PFTEMPLATE, param_sample, sim_name)

            # Save simulation outputs
            np.save(os.path.join(results_dir, f"ertt_{i}_stage_{learning_stage}.npy"), ertt)
            np.save(os.path.join(results_dir, f"ertd_{i}_stage_{learning_stage}.npy"), ertd)
            np.save(os.path.join(results_dir, f"obst_{i}_stage_{learning_stage}.npy"), obst)
            np.save(os.path.join(results_dir, f"obsd_{i}_stage_{learning_stage}.npy"), obsd)
            np.save(os.path.join(results_dir, f"params_{i}_stage_{learning_stage}.npy"), param_sample)

            # Clean up temporary files
            for file in os.listdir('.'):
                if 'run' in file and (file.endswith('.srv') or file.endswith('.pft')):
                    os.remove(file)
            
            # Move simulation files to output directory
            for file in os.listdir('.'):
                if 'run' in file and (file.endswith('.dat') or file.endswith('.chk') or file.endswith('.out')):
                    os.rename(file, os.path.join(sim_out_dir, file))
        print(f"PFLOTRAN simulations completed for stage {learning_stage}.")

    def plot_stage_progression(self, combined_params, field_params, stage=None):
        """
        Plot MDS projection showing progression of active learning stages.
        """
        DataVisualization.plot_mds_projection(
            data=combined_params, 
            field_data=field_params,
            labels=["Training Data", "Field Data", "Augmented Data"],
            title=f"MDS Projection Including Augmented Data (Stage {stage})"
        )

# Visualization Utilities
class DataVisualization:
    def __init__(self, config: Config, ActiveLearning: ActiveLearningLoop):
        self.config = config
        self.ActiveLearning = ActiveLearning

    def plot_mds_projection(self, data, field_data, label_data, labels, title="MDS Projection"):
        """
        Plot 2D MDS projections for the datasets.
        `data` includes the training, field, and augmented data.
        """
        mds = MDS(n_components=2, random_state=42)  # Reduce to 2 dimensions
        data_combined = np.vstack(data)
        embedded_data = mds.fit_transform(data_combined)

        # Split MDS projection into groups
        train_mds = embedded_data[:len(data[0])]
        field_mds = embedded_data[len(data[0]):len(data[0]) + len(data[1])]
        augmented_mds = embedded_data[len(data[0]) + len(data[1]):]

        # Scale for visualization
        scaler = MinMaxScaler()
        train_mds_scaled = scaler.fit_transform(train_mds)
        field_mds_scaled = scaler.transform(field_mds)
        augmented_mds_scaled = scaler.transform(augmented_mds)

        plt.figure(figsize=(12, 8), dpi=350)
        plt.title(title, fontsize=22)
        # Plot density contour for training data
        sns.kdeplot(x=train_mds_scaled[:, 0], y=train_mds_scaled[:, 1],
                    fill=True, cmap='Blues', levels=10, thresh=0)
        plt.scatter(train_mds_scaled[:, 0], train_mds_scaled[:, 1],
                    alpha=0.5, s=30, color='blue', label=labels[0])
        plt.scatter(field_mds_scaled[:, 0], field_mds_scaled[:, 1],
                    color='red', marker='*', s=200, label=labels[1])
        plt.scatter(augmented_mds_scaled[:, 0], augmented_mds_scaled[:, 1],
                    alpha=0.7, s=50, color='orange', label=labels[2])

        plt.xlabel('Normalized MDS Dimension 1', fontsize=16)
        plt.ylabel('Normalized MDS Dimension 2', fontsize=16)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        plt.show(block=False)

    @staticmethod
    def plot_training_validation_loss(train_losses, val_losses, save_path=None):
        """
        Plot training and validation loss curves.
        Args:
            train_losses: Training loss values
            val_losses: Validation loss values
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        #plt.plot(val_losses, label='Validation Loss', color='orange')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.yscale('log')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        plt.show(block=False)

    @staticmethod
    def plot_well_data(well_data, conductivity=False, normalized=False, sample_idx=0, save_path=None):
        """
        Plot well data (tracer concentration or conductivity) for 8 wells with 3 replicates each.
        
        Args:
            well_data: Data array with shape (8, 3, time_points) - 8 wells, 3 replicates, time series
            conductivity: Boolean flag indicating if data is conductivity (True) or tracer (False)
            normalized: Boolean flag indicating if data is normalized
            sample_idx: Sample index for plot title
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(16, 8), dpi=350)
        for i in range(8):
            plt.subplot(2, 4, i+1) 
            for rep in range(3):
                plt.plot(well_data[i, rep, :], label=f'W{i+1}_{rep+1}')
            plt.legend()
            plt.title(f'Well {i+1}, Sample {sample_idx}')
            plt.xlabel('Time (h)')
            plt.grid(alpha=0.3)
            
            # Set appropriate y-label based on data type and normalization
            if conductivity:
                if normalized:
                    plt.ylabel('Normalized Conductivity')
                else:
                    plt.ylabel('Conductivity')
            else:
                if normalized:
                    plt.ylabel('Normalized Tracer Concentration')
                else:
                    plt.ylabel('Tracer Concentration')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=350, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        plt.show(block=False)

    @staticmethod
    def plot_wsse_time_series(WSSE_sim, sorted_indices, best_n=3, save_path=None):
        """
        Plot WSSE time series for all simulations highlighting the best performers.
        
        Args:
            WSSE_sim: WSSE values per simulation and time step
            sorted_indices: Indices sorted by total WSSE (best first)
            best_n: Number of best simulations to highlight
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(8, 5.5), dpi=250)
        for sim in range(WSSE_sim.shape[0]):
            plt.plot(WSSE_sim[sim, :], color='gray', alpha=0.7, linewidth=1)
        colors = cm.viridis(np.linspace(0, 1, best_n))
        for i, idx in enumerate(sorted_indices[:best_n]):
            plt.plot(
                WSSE_sim[idx, :],
                color=colors[i],
                linewidth=1.5,
                label=f'Sim {idx}'
            )
        plt.xlabel('ERT Survey')
        plt.yscale('log')
        plt.ylabel('WSSE')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=best_n, frameon=True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        plt.show(block=False)

    @staticmethod
    def plot_best_simulations_comparison(inv_ertd_data, field_ert_true, sorted_indices, wsse_metric_func, 
                                       best_n=3, save_path=None):
        """
        Plot detailed comparison of best simulations including ERT images, scatter plots, and KDE.
        Args:
            inv_ertd_data: Inverted ERT data
            field_ert_true: Field ERT true data
            sorted_indices: Indices sorted by performance
            wsse_metric_func: WSSE metric function (uses config internally)
            best_n: Number of best simulations to plot
            save_path: Path to save the figure (optional)
        """
        universal_fontsize = 16
        fig, axs = plt.subplots(nrows=best_n, ncols=3, figsize=(18, 16), dpi=250)
        for row, idx in enumerate(sorted_indices[:best_n]):
            WSSE, _ = wsse_metric_func(inv_ertd_data[idx, :, :], field_ert_true)
            # ERT image
            im = axs[row, 0].imshow(inv_ertd_data[idx, :, :], aspect='auto', origin='lower', cmap='jet')
            axs[row, 0].set_title(f'Sim {idx}, WSSE={WSSE:.2f}', fontsize=universal_fontsize)
            axs[row, 0].set_xlabel("ERT Surveys", fontsize=universal_fontsize)
            axs[row, 0].set_ylabel("ERT Measurements", fontsize=universal_fontsize)
            axs[row, 0].tick_params(axis='both', which='major', labelsize=universal_fontsize)
            cbar = fig.colorbar(im, ax=axs[row, 0], fraction=0.046, pad=0.04)
            cbar.set_label('Transfer resistance [Ω]', fontsize=universal_fontsize)
            cbar.ax.tick_params(labelsize=universal_fontsize)
            # Scatter plot
            min_val = min(np.min(inv_ertd_data[idx, :, :]), np.min(field_ert_true))
            max_val = max(np.max(inv_ertd_data[idx, :, :]), np.max(field_ert_true))
            axs[row, 1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1, label='Identity Line')
            axs[row, 1].scatter(inv_ertd_data[idx, :, :].flatten(), field_ert_true.flatten(), color='black', s=5)
            axs[row, 1].set_xlabel('Inverted ERT [Ω]', fontsize=universal_fontsize)
            axs[row, 1].set_ylabel('Conditional ERT [Ω]', fontsize=universal_fontsize)
            axs[row, 1].legend(fontsize=universal_fontsize)
            axs[row, 1].set_aspect('equal', adjustable='box')
            axs[row, 1].grid(True, which='both', linestyle='--', linewidth=0.3)
            # Distribution comparison (KDE)
            sns.kdeplot(inv_ertd_data[idx, :, :].flatten(), color='C1', label='Inverted ERT', ax=axs[row, 2])
            sns.kdeplot(field_ert_true.flatten(), color='C0', label='Conditional ERT', ax=axs[row, 2])
            axs[row, 2].set_xlabel('Transfer resistance [Ω]', fontsize=universal_fontsize)
            axs[row, 2].set_ylabel('Density', fontsize=universal_fontsize)
            axs[row, 2].legend(fontsize=universal_fontsize)
            w_distance = wasserstein_distance(inv_ertd_data[idx, :, :].flatten(), field_ert_true.flatten())
            axs[row, 2].set_title(f'Wasserstein Distance: {w_distance:.4f}', fontsize=universal_fontsize)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        plt.show(block=False)

    @staticmethod
    def plot_individual_simulation_comparison(inv_ertd_data, field_ert_true, sorted_indices, wsse_metric_func, save_dir=None):
        """
        Plot individual comparison of each simulation with field data.
        Args:
            inv_ertd_data: Inverted ERT data
            field_ert_true: Field ERT true data
            sorted_indices: Indices sorted by performance
            wsse_metric_func: WSSE metric function (uses config internally)
            save_dir: Directory to save the figures (optional)
            
        Returns:
            list: WSSE values for each simulation
        """
        wsse_values = []
        for rank, i in enumerate(sorted_indices):
            WSSE, _ = wsse_metric_func(inv_ertd_data[i], field_ert_true)
            wsse_values.append(WSSE)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=250)

            im0 = axes[0].imshow(field_ert_true, cmap='jet', aspect='auto',origin='lower')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            axes[0].set_title('Field ERT Data')
            axes[0].set_xlabel('ERT Surveys')
            axes[0].set_ylabel('ERT Measurements')

            im1 = axes[1].imshow(inv_ertd_data[i], cmap='jet', aspect='auto',origin='lower')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            axes[1].set_title(f'Inverted ERT Sim: {i}, WSSE={WSSE:.2f}')
            axes[1].set_xlabel('ERT Surveys')
            axes[1].set_ylabel('ERT Measurements')

            limits = [
                min(field_ert_true.min(), inv_ertd_data[i].min()),
                max(field_ert_true.max(), inv_ertd_data[i].max())
            ]
            axes[2].plot([limits[0], limits[1]], [limits[0], limits[1]], color='red', linestyle='--', linewidth=1, label='Identity Line')
            axes[2].scatter(field_ert_true.flatten(), inv_ertd_data[i].flatten(), color='black', s=5)
            axes[2].set_xlabel('Field ERT Data')
            axes[2].set_ylabel('Inverted ERT Data')
            axes[2].set_title(f'Simulation rank: {rank+1}')
            axes[2].grid(alpha=0.3)

            fig.tight_layout()
            if save_dir:
                save_path = os.path.join(save_dir, f'individual_sim_rank_{rank+1}_idx_{i}.png')
                plt.savefig(save_path, dpi=250, bbox_inches='tight')
                if rank == 0:  # Only print once to avoid spam
                    print(f"Saving individual simulation plots to: {save_dir}")
            plt.show(block=False)
        return wsse_values

    @staticmethod
    def plot_wsse_ranking(wsse_values, ranks=[1, 5, 10, 20, 30, 40, 50], save_path=None):
        """
        Plot WSSE ranking showing performance of all simulations with highlighted ranks.
        Args:
            wsse_values: WSSE values for all simulations (sorted)
            ranks: Specific ranks to highlight
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(8,5), dpi=250)
        plt.scatter(
            [r-1 for r in ranks if r <= len(wsse_values)],
            [wsse_values[r-1] for r in ranks if r <= len(wsse_values)],
            color='red',
            marker="*",
            s=80,
            label='Selected Ranks',
            zorder=3
        )
        plt.plot(wsse_values, marker='o', linestyle='-', linewidth=1, markersize=5, label='WSSE per Simulation')
        plt.xlabel('Ranked Simulation Index')
        plt.ylabel('WSSE')
        plt.title('WSSE of Inverted ERT Data Simulations')
        plt.legend()
        plt.xticks(
            [r - 1 for r in ranks if r <= len(wsse_values)],
            [str(r) for r in ranks if r <= len(wsse_values)]
        )
        plt.grid(alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        plt.show(block=False)

    @staticmethod
    def plot_conductivity_comparison(field_conductivity_arranged, inv_conductivity_arranged_all):
        """
        Plot comparison of field conductivity data with inverted conductivity data for all wells.
        Args:
            field_conductivity_arranged: Field conductivity data arranged by wells
            inv_conductivity_arranged_all: All inverted conductivity data arranged by wells
        """
        sam_loc_colors = ['C0', 'C1', 'C2']
        for well in range(field_conductivity_arranged.shape[0]):
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=350)
            for sam_loc in range(field_conductivity_arranged.shape[1]):
                ax = axes[sam_loc]
                for td in range(inv_conductivity_arranged_all.shape[0]):
                    ax.plot(
                        inv_conductivity_arranged_all[td, well, sam_loc, :],
                        color=sam_loc_colors[sam_loc],
                        alpha=0.1,
                        linewidth=1,
                        label = 'Inverted' if td == 0 else None
                    )
                ax.plot(
                    field_conductivity_arranged[well, sam_loc, :],
                    linestyle='--',
                    color=sam_loc_colors[sam_loc],
                    linewidth=1,
                    label='Field'
                )
                ax.set_ylabel('Conductivity (S/m)')
                ax.set_title(f'Well {well+1} – Location {sam_loc+1}')
                ax.grid(alpha=0.3)
                if sam_loc == 0:
                    ax.legend(loc='upper right')
            axes[-1].set_xlabel('Time (h)')
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show(block=False)

#%% CDM Active Learning Workflow

# Set random seeds for reproducibility
set_random_seeds()

# Initialize configuration manager
config = Config()

# Initialize data handler
data_handler = DataLoaderHandler(config)

# Load data - automatically choose between NPZ file and individual files
npz_file_path = '/people/hern856/GIFT_Paralell_Uniform_Dataset/Sobol_simulated_data_300/sobol_augmented_data.npz'
ertt_data, ertd_data, param_data, conductivity_data, tracer_data = data_handler.load_data_auto(
    npz_file_path=npz_file_path, 
    total_simulations=300, 
    stage=None
)

# Establish all scalers first using the initial training data
data_scalers = data_handler.establish_scalers(param_data, ertd_data, tracer_data, conductivity_data)


# Normalize data using established scalers
ert_norm = data_handler.scale_ert_data(ertd_data)
tracer_norm = data_handler.scale_tracer_data(tracer_data)
conductivity_norm = data_handler.scale_conductivity_data(conductivity_data)
param_norm = data_handler.normalize_parameters(param_data)

#np.save(os.path.join(config.RESULTS_DIR, 'normalized_initial_param_data.npy'), param_norm)
#np.save(os.path.join(config.RESULTS_DIR, 'normalized_initial_ertd_data.npy'), ert_norm)
#np.save(os.path.join(config.RESULTS_DIR, 'normalized_initial_tracer_data.npy'), tracer_norm)
#np.save(os.path.join(config.RESULTS_DIR, 'normalized_initial_conductivity_data.npy'), conductivity_norm)


# Initialize CDM model
cdm = ConditionalDiffusionModel(
    ert_shape=ert_norm.shape,
    well_obs_shape=conductivity_norm.shape,
    param_dim=param_norm.shape[1]
)

dataset = ConditionalDataset(ert_data=ert_norm, tracer_data=tracer_norm, conductivity_data=conductivity_norm, parameter_data=param_norm)

field_size = 1 #Randomly select a sample as field data
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size - field_size

train_dataset, val_dataset, field_dataset = random_split(dataset, [train_size, val_size, field_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
field_loader = DataLoader(field_dataset, batch_size=1, shuffle=False)

# Initialize training history tracker
history_tracker = TrainingHistoryTracker(config)

# Initialize trainer of CDM
trainer = Trainer(cdm, {"train": train_loader, "val": val_loader}, config, history_tracker)

# Initialize Learning Rate Scheduler (ReduceLROnPlateau)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    trainer.optimizer,
    mode='min',           # Minimize loss
    factor=0.5,           # Reduce LR by 50%
    patience=2,           # Wait 2 stages before reducing
    threshold=0.01,       # 1% relative improvement threshold
    threshold_mode='rel',
    min_lr=1e-6,          # Minimum learning rate
    verbose=True          # Print when LR changes
)
print(f"[INFO] Learning rate scheduler initialized: ReduceLROnPlateau (patience=2, factor=0.5, min_lr=1e-6)")

# Determine if resuming from a previous stage
resume_stage = config.RESUME_FROM_STAGE

# Train CDM initially (Stage 0)
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "cdm_stage_0.pt")



# Active Learning Loop Implementation
learning_loop = ActiveLearningLoop(trainer, config, pf, data_handler)
num_stages = config.NUM_LEARNING_STAGES
augmented_datasets = []
augmented_val_datasets = []  # Track validation samples from each stage

# Field data for comparison and conditioning
field_ert, field_tracer, field_conductivity, field_params = next(iter(field_loader))
field_ert_npy_norm = field_ert.detach().cpu().numpy().squeeze()
field_tracer_npy_norm = field_tracer.detach().cpu().numpy().squeeze()
field_conductivity_npy_norm = field_conductivity.detach().cpu().numpy().squeeze()
print ('Field ERT shape:', field_ert_npy_norm.shape)
print ('Field Tracer shape:', field_tracer_npy_norm.shape)
print ('Field Conductivity shape:', field_conductivity_npy_norm.shape)

field_params_npy_norm = field_params.detach().cpu().numpy().squeeze()
field_params_true = data_handler.inverse_transform_parameters(field_params_npy_norm.reshape(1, -1))
print ('Field Parameters True shape:', field_params_true.shape)
np.save(os.path.join(config.RESULTS_DIR, 'field_params_true.npy'), field_params_true)

#Run simulation on true parameters

# Inverse transform field data to true scale and save
field_ert_flat = field_ert_npy_norm.reshape(1, -1)
field_ert_true = data_handler.inverse_transform_ert_data(field_ert_npy_norm)
print ('Field ERT True shape:', field_ert_true.shape)
np.save(os.path.join(config.RESULTS_DIR, 'field_ert_true.npy'), field_ert_true)

field_tracer_flat = field_tracer_npy_norm.reshape(1, -1)
field_tracer_true = data_handler.inverse_transform_tracer_data(field_tracer_npy_norm)
print ('Field Tracer True shape:', field_tracer_true.shape)
np.save(os.path.join(config.RESULTS_DIR, 'field_tracer_true.npy'), field_tracer_true)
field_tracer_arranged = data_handler.well_data_arrange(field_tracer_true,sample_idx=None)
save_path = os.path.join(config.RESULTS_DIR, 'field_tracer_data.png')
DataVisualization.plot_well_data(field_tracer_arranged,conductivity=False,normalized=False, sample_idx=None, save_path=save_path)

field_conductivity_flat = field_conductivity_npy_norm.reshape(1, -1)
field_conductivity_true = data_handler.inverse_transform_conductivity_data(field_conductivity_npy_norm)
print ('Field Conductivity True shape:', field_conductivity_true.shape)
np.save(os.path.join(config.RESULTS_DIR, 'field_conductivity_true.npy'), field_conductivity_true)
field_conductivity_arranged = data_handler.well_data_arrange(field_conductivity_true,sample_idx=None)
save_path = os.path.join(config.RESULTS_DIR, 'field_conductivity_data.png')
DataVisualization.plot_well_data(field_conductivity_arranged,conductivity=True,normalized=False, sample_idx=None, save_path=save_path)

#Verify that the expected simulation output files for the true parameters are present before attempting to train the model
learning_loop.pflotran_simulate_parallel(param_array=field_params_true, learning_stage=None, case_name='Test')
fpt_ertt, fpt_ertd, fpt_params, fpt_conductivity, fpt_tracer = data_handler.load_data(total_simulations=field_params_true.shape[0], stage=None, case_name='Test')

_, fpt_wsse_total, _ = learning_loop.evaluate_simulations_against_field(fpt_ertd, field_ert_true)


if fpt_wsse_total[0] < 1e-3:
    print(f"\n[INFO] Test simulation with true parameters has very low WSSE ({fpt_wsse_total[0]:.6f}), indicating good match with field data.")
    print(f"       This suggests that the PFLOTRAN simulation outputs are consistent with the field data for the true parameters.")
else:
    print(f"\n[ERROR] Test simulation with true parameters has high WSSE ({fpt_wsse_total[0]:.6f}), indicating poor match with field data.")
    print(f"        This suggests issues with the PFLOTRAN simulation outputs or the data loading process for the true parameters.")
    print(f"        Please verify that the simulation ran successfully and that the output files are correct.")
    print(f"\n[STOPPING] Halting execution due to field parameter verification failure.")
    sys.exit(1)


if resume_stage is None or resume_stage == 0:
    print(f"\n[INFO] Starting initial training of CDM (Stage 0)...")
    train_losses, val_losses = trainer.train(
        num_epochs=config.NUM_EPOCHS, 
        checkpoint_path=checkpoint_path, 
        continue_from_checkpoint=False,
        stage_name="Stage 0",
        num_augmented_samples=0
    )
    # Visualize losses for initial training
    save_path = os.path.join(config.RESULTS_DIR, 'training_validation_loss_stage_0.png')
    DataVisualization.plot_training_validation_loss(train_losses, val_losses, save_path=save_path)
else:
    print(f"\n[INFO] Resuming from Stage {resume_stage} - Skipping Stage 0 training...")
    print(f"[INFO] Stage 0 checkpoint should exist at: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cannot resume: Stage 0 checkpoint not found at {checkpoint_path}")

num_cdm_samples = 30

# Run Active Learning Stages
start_stage = resume_stage if resume_stage is not None else 1
for stage in range(start_stage, num_stages + 1):
    print(f"\n{'='*70}")
    print(f"Starting Active Learning Stage {stage}...")
    print(f"{'='*70}")

    # Sample New Realizations using Adaptive Sampling
    model_checkpoint = os.path.join(config.CHECKPOINT_DIR, f"cdm_stage_{stage - 1}.pt")
    
    trainer.load_model_only(model_checkpoint)
    print(f"[INFO] Loaded model from: {model_checkpoint}")

    wsse_guided_samples = learning_loop.wsse_guided_sampling(num_samples=num_cdm_samples)
    if wsse_guided_samples is not None:
        print(f"[INFO] Sampling mode: WSSE-guided archive (generated {len(wsse_guided_samples)} samples)")
        valid_sampled_params = wsse_guided_samples
    else:
        print(f"[INFO] Sampling mode: CDM clipping fallback (insufficient WSSE archive)")
        valid_sampled_params = learning_loop.sample_model_with_clipping(
            ert_data=field_ert,
            well_obs_data=field_conductivity,
            params_dim=param_data.shape[1],
            num_samples=num_cdm_samples,
            temperature=1.0,
            conditioning_noise_std=0.0,
            use_diverse_conditioning=False
        )
    
    # Save sampled parameters for potential resume
    sampled_params_file = os.path.join(config.RESULTS_DIR, f'sampled_params_stage_{stage}.npy')
    np.save(sampled_params_file, valid_sampled_params)
    print(f"       Saved sampled parameters to: {sampled_params_file}")

    random_state_file = os.path.join(config.RESULTS_DIR, f'random_state_stage_{stage}.pkl')

    import pickle
    random_state = {
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state()
    }
    with open(random_state_file, 'wb') as f:
        pickle.dump(random_state, f)
    print(f"       Saved random state to: {random_state_file}")
    
    learning_loop.pflotran_simulate_parallel(param_array=valid_sampled_params, learning_stage=stage,case_name="WSSE_guided")
    cdm_ertt, cdm_ertd, cdm_params, cdm_conductivity, cdm_tracer = data_handler.load_data(total_simulations=valid_sampled_params.shape[0], stage=stage, case_name="WSSE_guided")
    # Check if some simulations failed
    if cdm_ertd.shape[0] < valid_sampled_params.shape[0]:
        print(f"\n[WARNING] Some CDM simulations failed!")
        print(f"  Expected: {valid_sampled_params.shape[0]} simulations")
        print(f"  Loaded:   {cdm_ertd.shape[0]} simulations")
        print(f"  Failed:   {valid_sampled_params.shape[0] - cdm_ertd.shape[0]} simulations")
        print(f"  Continuing with {cdm_ertd.shape[0]} successful simulations...\n")
    
    # Scale CDM simulation data to match training data format
    cdm_conductivity_scaled = data_handler.scale_conductivity_data(cdm_conductivity)
    cdm_tracer_scaled = data_handler.scale_tracer_data(cdm_tracer)
    cdm_ert_scaled = data_handler.scale_ert_data(cdm_ertd)
    cdm_params_normalized = data_handler.normalize_parameters(cdm_params)
    cdm_params_normalized_all = cdm_params_normalized.copy()
    
    WSSE_sim, wsse_total, sorted_indices = learning_loop.evaluate_simulations_against_field(
        cdm_ertd, field_ert_true)
    
    # Report best performance against CLEAN field data
    best_wsse = wsse_total[sorted_indices[0]]
    p10_wsse = np.percentile(wsse_total, 10)
    p50_wsse = np.percentile(wsse_total, 50)
    p90_wsse = np.percentile(wsse_total, 90)

    #Save WSSE results
    wsse_results_file = os.path.join(config.RESULTS_DIR, f'wsse_results_stage_{stage}.npz')
    np.savez_compressed(
        wsse_results_file,
        WSSE_sim=WSSE_sim,
        wsse_total=wsse_total,
        sorted_indices=sorted_indices
    )
    print(f"       Saved WSSE results to: {wsse_results_file}")
    
    
    print(f"\n  CDM WSSE Statistics:")
    print(f"    Best (min):  {best_wsse:.4f}  [Simulation {sorted_indices[0]}]")
    print(f"    P10:         {p10_wsse:.4f}")
    print(f"    Median:      {p50_wsse:.4f}")
    print(f"    P90:         {p90_wsse:.4f}")
    print(f"    Worst (max): {wsse_total.max():.4f}")

    learning_loop.record_field_fit_metrics(stage, wsse_total, sorted_indices, cdm_params_normalized_all)
    

    # Plot WSSE time series
    save_path = os.path.join(config.RESULTS_DIR, f'wsse_time_series_stage_{stage}.png')
    DataVisualization.plot_wsse_time_series(WSSE_sim, sorted_indices, best_n=3, save_path=save_path)
    
    # Plot detailed comparison of best simulations
    save_path = os.path.join(config.RESULTS_DIR, f'best_WSSE_samples_comparison_stage_{stage}.png')
    DataVisualization.plot_best_simulations_comparison(
        cdm_ertd, field_ert_true, sorted_indices, learning_loop.wsse_metric, best_n=3, save_path=save_path)
    
    # Plot individual simulation comparisons
    save_dir = os.path.join(config.RESULTS_DIR, f'stage_{stage}_individual_sims')
    os.makedirs(save_dir, exist_ok=True)
    wsse_values = DataVisualization.plot_individual_simulation_comparison(
        cdm_ertd, field_ert_true, sorted_indices, learning_loop.wsse_metric, save_dir=save_dir
    )
    
    # Plot WSSE ranking
    save_path = os.path.join(config.RESULTS_DIR, f'wsse_ranking_stage_{stage}.png')
    DataVisualization.plot_wsse_ranking(wsse_values, save_path=save_path)

    print(f"\n[UNCERTAINTY FILTERING] Detecting and removing outlier uncertainty samples...")
    median_act = np.median(wsse_total)
    mad_act = np.median(np.abs(wsse_total - median_act))
    outlier_threshold_act = median_act + 9 * mad_act  # 9 MAD threshold
    
    valid_act_mask = wsse_total < outlier_threshold_act
    num_outliers_act = np.sum(~valid_act_mask)
    
    print(f"  Median WSSE: {median_act:.2f}")
    print(f"  MAD: {mad_act:.2f}")
    print(f"  Outlier threshold (median + 9*MAD): {outlier_threshold_act:.2f}")
    print(f"  Outliers detected: {num_outliers_act} / {len(wsse_total)}")
    
    if num_outliers_act > 0:
        print(f"  → Removing {num_outliers_act} extreme samples to prevent training instability")
        cdm_ert_scaled = cdm_ert_scaled[valid_act_mask]
        cdm_tracer_scaled = cdm_tracer_scaled[valid_act_mask]
        cdm_conductivity_scaled = cdm_conductivity_scaled[valid_act_mask]
        cdm_params_normalized = cdm_params_normalized[valid_act_mask]
        wsse_act_total_filtered = wsse_total[valid_act_mask]
        print(f"  → Retained {cdm_params_normalized.shape[0]} uncertainty samples for training")
    else:
        print(f"  → No outliers detected, using all {len(wsse_total)} uncertainty samples")
        wsse_act_total_filtered = wsse_total

    # Persist filtered samples/WSSE for future guided sampling
    learning_loop.update_wsse_archive(cdm_params_normalized, wsse_act_total_filtered)

    
    # Create training dataset from augmented samples (with refitted scalers)
    cdm_samples_dataset = ConditionalDataset(
        ert_data=cdm_ert_scaled, 
        tracer_data=cdm_tracer_scaled, 
        conductivity_data=cdm_conductivity_scaled, 
        parameter_data=cdm_params_normalized
    )

    augmented_datasets.append(cdm_samples_dataset)
    
    combined_train_dataset = ConcatDataset([train_dataset, *augmented_datasets])
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    trainer.data_loaders['train'] = combined_train_loader
    stage_checkpoint = os.path.join(config.CHECKPOINT_DIR, f"cdm_stage_{stage}.pt")
    previous_checkpoint = os.path.join(config.CHECKPOINT_DIR, f"cdm_stage_{stage - 1}.pt")
    
    print(f"[INFO] Continuing CDM training from stage {stage-1} with augmented dataset (Stage {stage})...")
    print(f"  Total training samples: {len(combined_train_dataset)}")

    stage_train_losses, stage_val_losses = trainer.continue_training_from_last_epoch(
         previous_checkpoint_path=previous_checkpoint,
         new_checkpoint_path=stage_checkpoint,
         num_epochs=config.NUM_EPOCHS,
         stage_name=f"Stage {stage}",
         num_augmented_samples=valid_sampled_params.shape[0])

    #Check model performance improvement after training with augmented data
    active_cdm_samples = learning_loop.sample_model_with_clipping(
    ert_data=field_ert,
    well_obs_data=field_conductivity,
    params_dim=param_data.shape[1],
    num_samples=10,
    temperature=1.0,
    conditioning_noise_std=0.0,
    use_diverse_conditioning=False)

    learning_loop.pflotran_simulate_parallel(param_array=active_cdm_samples, learning_stage=stage, case_name="Active_CDM")
    active_cdm_ertt, active_cdm_ertd, active_cdm_params, active_cdm_conductivity, active_cdm_tracer = data_handler.load_data(total_simulations=active_cdm_samples.shape[0], stage=stage, case_name="Active_CDM")
    
    active_WSSE_sim, active_wsse_total, active_WSSE_simsorted_indices = learning_loop.evaluate_simulations_against_field(
        active_cdm_ertd, field_ert_true)
    
    # Report best performance against CLEAN field data
    active_best_wsse = active_wsse_total[active_WSSE_simsorted_indices[0]]
    active_p10_wsse = np.percentile(active_wsse_total, 10)
    active_p50_wsse = np.percentile(active_wsse_total, 50)
    active_p90_wsse = np.percentile(active_wsse_total, 90)

    print(f"\n  Active CDM WSSE Statistics:")
    print(f"    Best (min):  {active_best_wsse:.4f}  [Simulation {active_WSSE_simsorted_indices[0]}]")
    print(f"    P10:         {active_p10_wsse:.4f}")
    print(f"    Median:      {active_p50_wsse:.4f}")
    print(f"    P90:         {active_p90_wsse:.4f}")
    print(f"    Worst (max): {active_wsse_total.max():.4f}")

    active_cdm_conductivity_scaled = data_handler.scale_conductivity_data(active_cdm_conductivity)
    active_cdm_tracer_scaled = data_handler.scale_tracer_data(active_cdm_tracer)
    active_cdm_ert_scaled = data_handler.scale_ert_data(active_cdm_ertd)
    active_cdm_params_normalized = data_handler.normalize_parameters(active_cdm_params)
    active_cdm_params_normalized_all = active_cdm_params_normalized.copy()  

    median_act = np.median(active_wsse_total)
    mad_act = np.median(np.abs(active_wsse_total - median_act))
    outlier_threshold_act = median_act + 9 * mad_act  # 9 MAD threshold
    
    valid_act_mask = active_wsse_total < outlier_threshold_act
    num_outliers_act = np.sum(~valid_act_mask)
    
    if num_outliers_act > 0:
        print(f"  → Removing {num_outliers_act} extreme samples to prevent training instability")
        active_cdm_ert_scaled = active_cdm_ert_scaled[valid_act_mask]
        active_cdm_tracer_scaled = active_cdm_tracer_scaled[valid_act_mask]
        active_cdm_conductivity_scaled = active_cdm_conductivity_scaled[valid_act_mask]
        active_cdm_params_normalized = active_cdm_params_normalized[valid_act_mask]
        wsse_act_total_filtered = active_wsse_total[valid_act_mask]
        print(f"  → Retained {active_cdm_params_normalized.shape[0]} uncertainty samples for training")
    else:
        print(f"  → No outliers detected, using all {len(active_wsse_total)} uncertainty samples")
        wsse_act_total_filtered = active_wsse_total

    learning_loop.update_wsse_archive(active_cdm_params_normalized, wsse_act_total_filtered)

        
    # Plot detailed comparison of active CDM predictions simulations
    save_path = os.path.join(config.RESULTS_DIR, f'active_WSSE_samples_comparison_stage_{stage}.png')
    DataVisualization.plot_best_simulations_comparison(
        active_cdm_ertd, field_ert_true, active_WSSE_simsorted_indices, learning_loop.wsse_metric, best_n=3, save_path=save_path)
    
    # Update learning rate based on training loss (ReduceLROnPlateau)
    final_train_loss = stage_train_losses[-1]
    lr_scheduler.step(final_train_loss)
    current_lr = trainer.optimizer.param_groups[0]['lr']
    print(f"[LR SCHEDULER] Stage {stage} final train loss: {final_train_loss:.6f}, LR: {current_lr:.2e}")

    # Visualize Losses for Current Stage
    save_path = os.path.join(config.RESULTS_DIR, f'training_validation_loss_stage_{stage}.png')
    DataVisualization.plot_training_validation_loss(stage_train_losses, stage_val_losses, save_path=save_path)


print(f"Saving complete training history across all {num_stages + 1} stages...")
history_tracker.save_history()

print(f"Plotting cumulative training history...")
history_tracker.plot_cumulative_history()

print(f"Active learning pipeline completed")
print(f"Total stages: {len(history_tracker.stage_info)}")
print(f"Total epochs: {history_tracker.current_epoch}")
print(f"Final validation loss: {history_tracker.cumulative_val_losses[-1]:.6f}")

