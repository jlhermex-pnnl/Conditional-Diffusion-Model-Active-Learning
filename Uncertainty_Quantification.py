import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import os
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

class Config:
    LOG_PARAMETERS = [2, 4, 7, 9, 12, 14]  # Parameters known to be log-ed

class DataLoaderHandler:
    """
    Handles data loading, scaling, and preprocessing with consistent scaler management.
    Args:
        config: Configuration object with data directory and other settings
    """
    def __init__(self, config: Config):
        self.config = config
        self.parameter_scalers = {}

    def establish_scalers(self, parameter_data_array, ert_array, tracer_array, cond_array):  
        self._reference_ert_shape = ert_array.shape
        self.parameter_scalers = {}
        for i in range(parameter_data_array.shape[1]):
            scaler = MinMaxScaler()
            
            # Apply log transformation if this is a log parameter
            if i in self.config.LOG_PARAMETERS:
                param_data = np.log10(parameter_data_array[:, i])
            else:
                param_data = parameter_data_array[:, i]
            scaler.fit(param_data.reshape(-1, 1))
            self.parameter_scalers[i] = scaler
        self.ert_scaler = StandardScaler()
        self.ert_scaler.fit(ert_array.reshape(-1, 1))
        self.tracer_scaler = StandardScaler()
        self.tracer_scaler.fit(tracer_array.reshape(-1, 1))
        self.conductivity_scaler = StandardScaler()
        self.conductivity_scaler.fit(cond_array.reshape(-1, 1))

    def normalize_parameters(self, parameter_data_array):
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
        if not hasattr(self, 'ert_scaler'):
            raise ValueError("ERT scaler not established. Call establish_scalers() first.")
        scaled_data = self.ert_scaler.transform(ert_data.reshape(-1, 1)).reshape(ert_data.shape)
        return scaled_data
    def scale_tracer_data(self, tracer_data):
        if not hasattr(self, 'tracer_scaler'):
            raise ValueError("Tracer scaler not established. Call establish_scalers() first.")
        scaled_data = self.tracer_scaler.transform(tracer_data.reshape(-1, 1)).reshape(tracer_data.shape)
        return scaled_data
    def scale_conductivity_data(self, conductivity_data):
        if not hasattr(self, 'conductivity_scaler'):
            raise ValueError("Conductivity scaler not established. Call establish_scalers() first.")
        scaled_data = self.conductivity_scaler.transform(conductivity_data.reshape(-1, 1)).reshape(conductivity_data.shape)
        return scaled_data
    def inverse_transform_parameters(self, scaled_params):
        if not hasattr(self, 'parameter_scalers') or not self.parameter_scalers:
            raise ValueError("Parameter scalers not established. Call establish_scalers() first.")
        if scaled_params.ndim == 1:  # Single sample
            params_original = np.zeros_like(scaled_params)
            for i in range(len(scaled_params)):
                param_value = self.parameter_scalers[i].inverse_transform([[scaled_params[i]]])[0, 0]
                if i in self.config.LOG_PARAMETERS:
                    params_original[i] = 10 ** param_value  # 10^x to reverse log10
                else:
                    params_original[i] = param_value
        else:  # Multiple samples
            params_original = np.zeros_like(scaled_params)
            for i in range(scaled_params.shape[1]):
                param_values = self.parameter_scalers[i].inverse_transform(scaled_params[:, i].reshape(-1, 1)).flatten()
                if i in self.config.LOG_PARAMETERS:
                    params_original[:, i] = 10 ** param_values  # 10^x to reverse log10
                else:
                    params_original[:, i] = param_values
        return params_original

    def inverse_transform_ert_data(self, scaled_ert_data):
        if not hasattr(self, 'ert_scaler'):
            raise ValueError("ERT scaler not established. Call establish_scalers() first.")
        original_shape = scaled_ert_data.shape
        ert_original = self.ert_scaler.inverse_transform(scaled_ert_data.reshape(-1, 1)).reshape(original_shape)
        return ert_original

    def inverse_transform_tracer_data(self, scaled_tracer_data):
        if not hasattr(self, 'tracer_scaler'):
            raise ValueError("Tracer scaler not established. Call establish_scalers() first.")
        original_shape = scaled_tracer_data.shape
        tracer_original = self.tracer_scaler.inverse_transform(scaled_tracer_data.reshape(-1, 1)).reshape(original_shape)
        return tracer_original

    def inverse_transform_conductivity_data(self, scaled_conductivity_data):
        if not hasattr(self, 'conductivity_scaler'):
            raise ValueError("Conductivity scaler not established. Call establish_scalers() first.")
        original_shape = scaled_conductivity_data.shape
        conductivity_original = self.conductivity_scaler.inverse_transform(scaled_conductivity_data.reshape(-1, 1)).reshape(original_shape)
        return conductivity_original

def well_data_arrange(tracer_cont_data, sample_idx=None):
    """
    Rearrange tracer/conductivity data from flat well format to structured format.
    
    Args:
        tracer_cont_data: Array of shape (n_samples, 24, 73) or (24, 73)
                            24 = 8 wells × 3 replicates, 73 = time points
        sample_idx: Sample index to extract. If None, processes all samples.
    
    Returns:
        tracer_arranged: Array of shape (n_samples, 8, 3, 73) or (8, 3, 73)
                        [samples, wells, replicates, time] or [wells, replicates, time]
    """
    # Determine if input is 2D or 3D
    if tracer_cont_data.ndim == 2:
        # Input is (24, 73) - single sample
        tracer_arranged = np.zeros((8, 3, 73))
        for w in range(8):
            for rep in range(3):
                tracer_arranged[w, rep, :] = tracer_cont_data[w*3+rep, :]
    elif tracer_cont_data.ndim == 3:
        # Input is (n_samples, 24, 73) - multiple samples
        n_samples = tracer_cont_data.shape[0]
        if sample_idx is None:
            # Process all samples
            tracer_arranged = np.zeros((n_samples, 8, 3, 73))
            for s in range(n_samples):
                for w in range(8):
                    for rep in range(3):
                        tracer_arranged[s, w, rep, :] = tracer_cont_data[s, w*3+rep, :]
        else:
            # Process single sample
            tracer_arranged = np.zeros((8, 3, 73))
            for w in range(8):
                for rep in range(3):
                    tracer_arranged[w, rep, :] = tracer_cont_data[sample_idx, w*3+rep, :]
    else:
        raise ValueError(f"Expected 2D or 3D input, got shape {tracer_cont_data.shape}")
    
    return tracer_arranged

def calculate_mse_well_data(field_data_arranged, samples_data_arranged):
    """
    Calculate Mean Squared Error between field data and arranged sample data.
    
    Args:
        field_data_arranged: Array of shape (8, 3, 73) - field/true data
                            [wells, replicates, time]
        samples_data_arranged: Array of shape (n_samples, 8, 3, 73) or (8, 3, 73)
                              Sample predictions to compare
    
    Returns:
        mse_values: Array of shape (n_samples,) with MSE for each sample
                   If input is single sample (8, 3, 73), returns scalar
    """
    if samples_data_arranged.ndim == 3:
        # Single sample case: (8, 3, 73)
        mse = np.mean((samples_data_arranged - field_data_arranged) ** 2)
        return mse
    elif samples_data_arranged.ndim == 4:
        # Multiple samples case: (n_samples, 8, 3, 73)
        n_samples = samples_data_arranged.shape[0]
        mse_values = np.zeros(n_samples)
        for i in range(n_samples):
            mse_values[i] = np.mean((samples_data_arranged[i] - field_data_arranged) ** 2)
        return mse_values
    else:
        raise ValueError(f"Expected 3D or 4D input for samples, got shape {samples_data_arranged.shape}")


def plot_mds_projection(parameters_dict, stages_to_include=None, title="MDS Projection of Parameters Across Stages", 
                        print_diagnostics=False):
    """
    Plot 2D MDS projections for specified stages in the parameters dictionary.
    The function automatically handles all stages and highlights the field parameters.
    Stage 0 (baseline) is plotted in gray, while active learning stages use the inferno_r palette.
    Parameters:
    -----------
    parameters_dict : dict
        Dictionary containing parameter arrays for different stages.
        Expected keys: 'Field', 'stage_0', 'stage_1', 'stage_2', etc.
    stages_to_include : list, optional
        List of stage numbers to include. If None, plots all stages.
    title : str
        Title for the plot
    print_diagnostics : bool, optional
        If True, prints diagnostic information about parameter distances
    """
    field_data = parameters_dict['Field']
    all_stage_keys = sorted([k for k in parameters_dict.keys() if k.startswith('stage_')],
                           key=lambda x: int(x.split('_')[1]))
    
    # Filter stages if specified
    if stages_to_include is not None:
        stage_keys = [f"stage_{stage}" for stage in stages_to_include if f"stage_{stage}" in parameters_dict]
    else:
        stage_keys = all_stage_keys

    # Separate stage 0 from active learning stages
    has_stage_0 = 'stage_0' in stage_keys
    active_learning_stages = [k for k in stage_keys if k != 'stage_0']
    
    all_data = [parameters_dict[key] for key in stage_keys]
    all_data.append(field_data)
    data_combined = np.vstack(all_data)
    
    # DIAGNOSTIC: Print parameter space distances
    if print_diagnostics:
        print("\n" + "="*80)
        print("PARAMETER SPACE DIAGNOSTICS")
        print("="*80)
        print(f"Field parameters shape: {field_data.shape}")
        print(f"Field parameters: {field_data[0]}")
        for key in stage_keys:
            stage_params = parameters_dict[key]
            # Calculate Euclidean distances from each sample to field
            distances = np.linalg.norm(stage_params - field_data, axis=1)
            print(f"\n{key}:")
            print(f"  Samples: {len(stage_params)}")
            print(f"  Min distance to field: {distances.min():.6f}")
            print(f"  Mean distance to field: {distances.mean():.6f}")
            print(f"  Max distance to field: {distances.max():.6f}")
            print(f"  Closest sample params: {stage_params[distances.argmin()]}")
        print("="*80 + "\n")

    mds = MDS(n_components=2, random_state=42, n_init=10, max_iter=300)
    embedded_data = mds.fit_transform(data_combined)
    scaler = MinMaxScaler()
    embedded_data_scaled = scaler.fit_transform(embedded_data)
    
    stage_data_mds = {}
    start_idx = 0
    for key in stage_keys:
        end_idx = start_idx + len(parameters_dict[key])
        stage_data_mds[key] = embedded_data_scaled[start_idx:end_idx]
        start_idx = end_idx
    
    field_mds = embedded_data_scaled[start_idx:]
    
    fig = plt.figure(figsize=(8.8, 8), dpi=300)
    ax = plt.gca()
    
    # Use inferno_r palette only for active learning stages
    palette = sns.color_palette("inferno_r", len(active_learning_stages))
    
    # Plot stage 0 first with gray color (with label)
    if has_stage_0:
        ax.scatter(stage_data_mds['stage_0'][:, 0], stage_data_mds['stage_0'][:, 1],
                   alpha=0.6, s=40, color='gray', label='Uniform Samples')
    
    # Plot active learning stages with color palette (without labels)
    for idx, key in enumerate(active_learning_stages):
        stage_num = key.split('_')[1]
        color = palette[idx]
        ax.scatter(stage_data_mds[key][:, 0], stage_data_mds[key][:, 1],
                   alpha=0.6, s=40, color=color)
    
    ax.scatter(field_mds[:, 0], field_mds[:, 1],
               color='red', marker='*', s=300, label='Field', linewidths=2, zorder=10)
    ax.set_xlabel('Normalized MDS Dimension 1', fontsize=16)
    ax.set_ylabel('Normalized MDS Dimension 2', fontsize=16)
    ax.set_title(title, fontsize=22)
    ax.legend(fontsize=14, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Add categorical colorbar for active learning stages
    if len(active_learning_stages) > 0:
        stage_nums = [int(k.split('_')[1]) for k in active_learning_stages]
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cmap = ListedColormap(palette)
        bounds = np.arange(len(stage_nums) + 1)
        norm = BoundaryNorm(bounds, cmap.N)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                          cax=cbar_ax, ticks=np.arange(len(stage_nums)) + 0.5)
        cb.set_label('Active learning stage', fontsize=14, rotation=270, labelpad=20)
        cb.ax.set_yticklabels([f'{stage}' for stage in stage_nums], fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.91, 1])  # Leave space for colorbar
    plt.show()


def plot_mds_projection_ert(ert_datasets_dict, field_ert, stages_to_include=None, 
                           title="MDS Projection of ERT Predictions Across Stages", 
                           print_diagnostics=False, show_zoom=True, zoom_xlim=None, zoom_ylim=None):
    """
    Plot 2D MDS projections for ERT predictions at specified stages.
    Shows how ERT predictions cluster in observation space compared to field data.
    Stage 0 (uniform samples) is plotted in gray, active learning stages use inferno_r palette.
    
    Parameters:
    -----------
    ert_datasets_dict : dict
        Dictionary containing ERT arrays for different stages.
        Expected keys: 'Field', 'stage_0', 'stage_1', 'stage_2', etc.
        Each value should be array of shape (n_simulations, n_measurements, n_surveys)
    field_ert : np.ndarray
        Field ERT data of shape (n_measurements, n_surveys)
    stages_to_include : list, optional
        List of stage numbers to include. If None, plots all stages (excluding Field).
    title : str
        Title for the plot
    print_diagnostics : bool, optional
        If True, prints diagnostic information about ERT distances
    show_zoom : bool, optional
        If True, creates a two-panel plot with full view and zoomed view
    zoom_xlim : tuple, optional
        X-axis limits for zoom (min, max). If None, auto-calculated from data
    zoom_ylim : tuple, optional
        Y-axis limits for zoom (min, max). If None, auto-calculated from data
    """
    all_stage_keys = sorted([k for k in ert_datasets_dict.keys() if k.startswith('stage_')],
                           key=lambda x: int(x.split('_')[1]))
    
    # Filter stages if specified
    if stages_to_include is not None:
        stage_keys = [f"stage_{stage}" for stage in stages_to_include if f"stage_{stage}" in ert_datasets_dict]
    else:
        stage_keys = all_stage_keys
    
    # Separate stage_0 from active learning stages
    has_stage_0 = 'stage_0' in stage_keys
    active_learning_stages = [k for k in stage_keys if k != 'stage_0']
    
    # Flatten ERT data for MDS (each sample becomes a 1D vector)
    all_data = []
    stage_sample_counts = []
    field_flat = field_ert.flatten().reshape(1, -1)
    
    print(f"\nProcessing ERT data for MDS...")
    total_filtered = 0
    
    for key in stage_keys:
        ert_stage = ert_datasets_dict[key]  # Shape varies by stage
        original_shape = ert_stage.shape
        print(f"  {key}: original shape = {original_shape}")
        
        # Ensure consistent shape: (n_samples, n_measurements, n_surveys)
        # Field is (1272, 46) = (measurements, surveys)
        expected_measurements, expected_surveys = field_ert.shape
        
        if len(ert_stage.shape) == 3:
            n_samples = ert_stage.shape[0]
            # Check if dimensions match expected (measurements=1272, surveys=46)
            if ert_stage.shape[1] == expected_measurements and ert_stage.shape[2] == expected_surveys:
                # Already correct: (n_samples, 1272, 46)
                pass
            elif ert_stage.shape[1] == expected_surveys and ert_stage.shape[2] == expected_measurements:
                # Need to transpose: (n_samples, 46, 1272) -> (n_samples, 1272, 46)
                print(f"    → Transposing to match field shape")
                ert_stage = ert_stage.transpose(0, 2, 1)
            else:
                print(f"    WARNING: Shape mismatch! Expected measurements={expected_measurements}, surveys={expected_surveys}")
        
        # Flatten each simulation's ERT into 1D
        ert_flat = ert_stage.reshape(ert_stage.shape[0], -1)
        
        all_data.append(ert_flat)
        stage_sample_counts.append(len(ert_flat))
        print(f"    → Final: {len(ert_flat)} samples, shape = {ert_flat.shape}")
    
    if total_filtered > 0:
        print(f"\nTotal outliers filtered: {total_filtered}")
    
    # Add field data
    print(f"  Field: shape = {field_ert.shape}, flattened = {field_flat.shape}")
    all_data.append(field_flat)
    
    data_combined = np.vstack(all_data)
    print(f"\nCombined data shape: {data_combined.shape}")
    print(f"Total samples: {sum(stage_sample_counts)} + 1 (field) = {data_combined.shape[0]}")
    
    # DIAGNOSTIC: Print ERT space distances
    if print_diagnostics:
        print("\n" + "="*80)
        print("ERT SPACE DIAGNOSTICS")
        print("="*80)
        print(f"Field ERT shape: {field_ert.shape}")
        print(f"Field ERT flattened shape: {field_flat.shape}")
        for key in stage_keys:
            ert_stage = ert_datasets_dict[key]
            # Calculate Euclidean distances from each sample to field
            ert_flat = ert_stage.reshape(ert_stage.shape[0], -1)
            distances = np.linalg.norm(ert_flat - field_flat, axis=1)
            print(f"\n{key}:")
            print(f"  Samples: {len(ert_stage)}")
            print(f"  ERT shape per sample: {ert_stage.shape[1:]}")
            print(f"  Min distance to field: {distances.min():.6f}")
            print(f"  Mean distance to field: {distances.mean():.6f}")
            print(f"  Max distance to field: {distances.max():.6f}")
            print(f"  Closest sample index: {distances.argmin()}")
        print("="*80 + "\n")
    
    # Apply MDS
    print(f"\nApplying MDS to {data_combined.shape[0]} samples with {data_combined.shape[1]} features...")
    mds = MDS(n_components=2, random_state=42, n_init=10, max_iter=300)
    embedded_data = mds.fit_transform(data_combined)
    print(f"MDS stress: {mds.stress_:.2f}")
    
    scaler = MinMaxScaler()
    embedded_data_scaled = scaler.fit_transform(embedded_data)
    
    # Split back into stage groups
    stage_data_mds = {}
    start_idx = 0
    for i, key in enumerate(stage_keys):
        end_idx = start_idx + stage_sample_counts[i]
        stage_data_mds[key] = embedded_data_scaled[start_idx:end_idx]
        start_idx = end_idx
    
    field_mds = embedded_data_scaled[start_idx:]
    
    # Collect WSSE values for all samples in MDS space (for contour plotting)
    all_mds_points = []
    all_wsse_values = []
    
    if wsse_results is not None:
        for key in stage_keys:
            if key in stage_data_mds and key in wsse_results:
                all_mds_points.append(stage_data_mds[key])
                all_wsse_values.append(wsse_results[key])
        
        if len(all_mds_points) > 0:
            all_mds_points = np.vstack(all_mds_points)
            all_wsse_values = np.concatenate(all_wsse_values)
            print(f"\nWSSE contours: {len(all_wsse_values)} samples with WSSE values")
    
    # Calculate zoom limits if not provided - centered on field ERT
    if show_zoom and zoom_xlim is None and zoom_ylim is None:
        # Center on field position
        field_x, field_y = field_mds[0, 0], field_mds[0, 1]
        
        # Get bounds from active learning stages (exclude stage_0)
        all_active_points = []
        for key in active_learning_stages:
            if key in stage_data_mds:
                all_active_points.append(stage_data_mds[key])
        
        if len(all_active_points) > 0:
            all_active = np.vstack(all_active_points)
            
            # Calculate distances from field
            dist_x = np.abs(all_active[:, 0] - field_x)
            dist_y = np.abs(all_active[:, 1] - field_y)
            
            # Use 85th percentile to focus on core cluster, with minimum of 0.005
            extent_x = max(np.percentile(dist_x, 85), 0.005)
            extent_y = max(np.percentile(dist_y, 85), 0.005)
            
            # Use the larger extent for equal spacing, add 10% padding
            extent = max(extent_x, extent_y) * 1.1
            
            # Cap extent to reasonable zoom level
            extent = min(extent, 0.15)  # Maximum zoom extent
            
            # Center the zoom on field position with equal extent
            zoom_xlim = (max(0, field_x - extent), min(1, field_x + extent))
            zoom_ylim = (max(0, field_y - extent), min(1, field_y + extent))
    
    # Plotting
    if show_zoom:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8), dpi=300)
        axes = [ax1, ax2]
    else:
        fig = plt.figure(figsize=(8.8, 8), dpi=300)
        axes = [plt.gca()]
    
    # Use inferno_r palette only for active learning stages
    palette = sns.color_palette("inferno_r", len(active_learning_stages))
    
    for ax_idx, ax in enumerate(axes):
        # Plot stage 0 first with gray color (with label)
        if has_stage_0:
            ax.scatter(stage_data_mds['stage_0'][:, 0], stage_data_mds['stage_0'][:, 1],
                       alpha=0.6, s=40, color='gray', label='Uniform Samples')
        
        # Plot active learning stages with color palette (without individual labels)
        for idx, key in enumerate(active_learning_stages):
            stage_num = key.split('_')[1]
            color = palette[idx]
            ax.scatter(stage_data_mds[key][:, 0], stage_data_mds[key][:, 1],
                       alpha=0.6, s=40, color=color)
        
        # Plot field data
        ax.scatter(field_mds[:, 0], field_mds[:, 1],
                   color='red', marker='*', s=300, label='Field ERT', linewidths=2, zorder=10)
        
        ax.set_xlabel('Normalized MDS Dimension 1', fontsize=16)
        ax.set_ylabel('Normalized MDS Dimension 2', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Apply zoom to second plot
        if show_zoom and ax_idx == 1 and zoom_xlim is not None and zoom_ylim is not None:
            ax.set_xlim(zoom_xlim)
            ax.set_ylim(zoom_ylim)
            ax.set_title('Zoomed View', fontsize=18, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=14, loc='best')
        else:
            ax.legend(fontsize=14, loc='best')
            ax.set_title(title if ax_idx == 0 else title, fontsize=22 if not show_zoom else 18)
            
    # Add categorical colorbar for active learning stages
    if len(active_learning_stages) > 0:
        if show_zoom:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        else:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        
        cmap = ListedColormap(palette)
        bounds = np.arange(len(active_learning_stages) + 1)
        norm = BoundaryNorm(bounds, cmap.N)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                          cax=cbar_ax, ticks=np.arange(len(active_learning_stages)) + 0.5)
        cb.set_label('Active learning stage', fontsize=14, rotation=270, labelpad=20)
        stage_nums = [int(k.split('_')[1]) for k in active_learning_stages]
        cb.ax.set_yticklabels([f'{stage}' for stage in stage_nums], fontsize=12)
    
    if show_zoom:
        plt.tight_layout(rect=[0, 0, 0.91, 1])
    else:
        plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.show()


def plot_simulations_comparisons(inv_ertd_data, field_ert_true, sorted_indices, wsse_metric_func):
    for rank,idx in enumerate(sorted_indices):
        WSSE, _ = wsse_metric_func(inv_ertd_data[idx, :, :], field_ert_true)
        diff = inv_ertd_data[idx, :, :] - field_ert_true
        percentage_error = np.abs((diff / field_ert_true) * 100)
        # Plot field ERT, inverted ERT, difference, and percentage error
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), dpi=250)
        #Att title with WSSE and simulation index
        im0 = axs[0].imshow(field_ert_true, aspect='auto', origin='lower', cmap='jet')
        axs[0].set_title('Conditional ERT', fontsize=14)
        cbar0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        cbar0.set_label('Transfer resistance [Ω]', fontsize=12)
        im1 = axs[1].imshow(inv_ertd_data[idx, :, :], aspect='auto', origin='lower', cmap='jet')
        axs[1].set_title(f'Inverted ERT', fontsize=14)
        cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        cbar1.set_label('Transfer resistance [Ω]', fontsize=12)
        im3 = axs[2].imshow(percentage_error, aspect='auto', origin='lower', cmap='afmhot', vmin=0)
        axs[2].set_title(f'Rank {rank} - WSSE: {WSSE:.2f}', fontsize=14)
        cbar3 = fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)
        cbar3.set_label('Percentage Error [%]', fontsize=12)
        plt.tight_layout()
        plt.show(block=False)   

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
    fig, axs = plt.subplots(nrows=best_n, ncols=3, figsize=(18, 17), dpi=350)
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
        r2 = r2_score(field_ert_true.flatten(), inv_ertd_data[idx, :, :].flatten())
        axs[row, 1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1, label='Identity Line')
        axs[row, 1].scatter(inv_ertd_data[idx, :, :].flatten(), field_ert_true.flatten(), color='black', s=5, label=f'R²={r2:.4f}')
        axs[row, 1].set_xlabel('Inverted ERT [Ω]', fontsize=universal_fontsize)
        axs[row, 1].set_ylabel('Conditional ERT [Ω]', fontsize=universal_fontsize)
        #axs[row, 1].set_xscale('log')
        #axs[row, 1].set_yscale('log')
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
        plt.savefig(save_path, dpi=350, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    plt.show(block=False)


def plot_mds_per_stage(parameters_dict, stages_to_include=None):
    """
    Create individual MDS plots showing cumulative stages.
    Each plot shows stages from stage_0 up to the current stage, with overlap probability.
    The field location is fixed across all plots for consistent orientation.
    
    Parameters:
    -----------
    parameters_dict : dict
        Dictionary containing parameter arrays for different stages.
    stages_to_include : list, optional
        List of stage numbers to plot. If None, plots all stages.
    """
    
    field_data = parameters_dict['Field']
    all_stage_keys = sorted([k for k in parameters_dict.keys() if k.startswith('stage_')],
                            key=lambda x: int(x.split('_')[1]))
    
    if stages_to_include is None:
        stages_to_plot = [int(k.split('_')[1]) for k in all_stage_keys]
    else:
        stages_to_plot = stages_to_include
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Compute reference MDS with all stages for alignment
    all_data_ref = [parameters_dict[key] for key in all_stage_keys]
    all_data_ref.append(field_data)
    data_combined_ref = np.vstack(all_data_ref)
    
    mds_ref = MDS(n_components=2, random_state=42, n_init=10, max_iter=300)
    embedded_ref = mds_ref.fit_transform(data_combined_ref)
    scaler_ref = MinMaxScaler()
    embedded_ref_scaled = scaler_ref.fit_transform(embedded_ref)
    
    # Get reference field location
    field_ref_location = embedded_ref_scaled[-1:]
    
    # Create subplot grid
    n_plots = len(stages_to_plot)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), dpi=150)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for plot_idx, target_stage in enumerate(stages_to_plot):
        ax = axes[plot_idx]
        
        # Get stages from 0 to target_stage
        stage_keys = [k for k in all_stage_keys if int(k.split('_')[1]) <= target_stage]
        
        # Combine data for MDS
        all_data = [parameters_dict[key] for key in stage_keys]
        all_data.append(field_data)
        data_combined = np.vstack(all_data)
        
        # Compute MDS
        mds = MDS(n_components=2, random_state=42, n_init=10, max_iter=300)
        embedded_data = mds.fit_transform(data_combined)
        scaler = MinMaxScaler()
        embedded_data_scaled = scaler.fit_transform(embedded_data)
        
        # Get current field location
        field_mds_current = embedded_data_scaled[-1:]
        
        # Align current MDS to reference by translating to match field location
        translation = field_ref_location - field_mds_current
        embedded_data_aligned = embedded_data_scaled + translation
        
        # Split back into groups
        stage_data_mds = {}
        start_idx = 0
        for key in stage_keys:
            end_idx = start_idx + len(parameters_dict[key])
            stage_data_mds[key] = embedded_data_aligned[start_idx:end_idx]
            start_idx = end_idx
        field_mds = embedded_data_aligned[start_idx:]
        
        # Calculate overlap KDE using aligned data
        x_min, x_max = embedded_data_aligned[:, 0].min() - 0.1, embedded_data_aligned[:, 0].max() + 0.1
        y_min, y_max = embedded_data_aligned[:, 1].min() - 0.1, embedded_data_aligned[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        grid_points = np.vstack([xx.ravel(), yy.ravel()])
        
        stage_kdes = []
        for key in stage_keys:
            if len(stage_data_mds[key]) > 1:
                kde = gaussian_kde(stage_data_mds[key].T)
                density = kde(grid_points)
                stage_kdes.append(density.reshape(xx.shape))
        
        # Plot overlap
        if len(stage_kdes) > 1:
            overlap_density = np.minimum.reduce(stage_kdes)
            contour = ax.contourf(xx, yy, overlap_density, levels=15, cmap='Greys', alpha=0.4)
        
        # Plot each stage
        for idx, key in enumerate(stage_keys):
            stage_num = key.split('_')[1]
            color = colors[idx % len(colors)]
            ax.scatter(stage_data_mds[key][:, 0], stage_data_mds[key][:, 1],
                      alpha=0.6, s=30, color=color, label=f'Stage {stage_num}')
        
        # Plot field
        ax.scatter(field_mds[:, 0], field_mds[:, 1],
                  color='red', marker='*', s=200, label='Field', 
                  linewidths=1.5, zorder=10)
        
        ax.set_xlabel('MDS Dimension 1', fontsize=12)
        ax.set_ylabel('MDS Dimension 2', fontsize=12)
        ax.set_title(f'Stages 0-{target_stage}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def wsse_metric( predictions, observations):
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
    sd = ALPHA_SD * np.abs(observations) + BETA_SD
    WSE = (predictions - observations)**2 / (sd)**2
    WSSE = np.average(WSE)
    return WSSE, WSE

def evaluate_simulations_against_field(cdm_ertd, field_ert_true):
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
            WSSE, _ = wsse_metric(cdm_ertd[sim][:, es], field_ert_true[:, es])
            WSSE_time.append(WSSE)
        WSSE_sim.append(WSSE_time)
    WSSE_sim = np.array(WSSE_sim)  

    wsse_total = WSSE_sim.sum(axis=1)
    sorted_indices = np.argsort(wsse_total)

    wsse_global = []
    for i in range(len(cdm_ertd)):
        wsse, _ = wsse_metric(cdm_ertd[i], field_ert_true)
        wsse_global.append(wsse)
    wsse_global = np.array(wsse_global)

    return WSSE_sim, wsse_global, sorted_indices

def plot_wsse_boxplot(wsse_results, title="WSSE Distribution Across Stages"):
    """
    Create box plots for WSSE total values across stages with a line connecting means.
    Parameters:
    -----------
    wsse_results : dict
        Dictionary containing WSSE total values for each stage.
        Expected keys: 'stage_1', 'stage_2', etc.
    title : str
        Title for the plot
    """
    stage_keys = sorted([k for k in wsse_results.keys() if k.startswith('stage_')],
                       key=lambda x: int(x.split('_')[1]))
    
    data_for_plot = []
    stage_labels = []
    
    for key in stage_keys:
        stage_num = key.split('_')[1]
        wsse_values = wsse_results[key]
        
        # Add data for box plot (as list)
        data_for_plot.append(wsse_values)
        stage_labels.append(f'Stage {stage_num}')

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    bp = ax.boxplot(data_for_plot, patch_artist=True, labels=stage_labels,
                    boxprops=dict(facecolor='lightblue', color='blue', linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='blue', linewidth=1.5),
                    capprops=dict(color='blue', linewidth=1.5),
                    flierprops=dict(markerfacecolor='blue', marker='o', markersize=5, 
                                   markeredgecolor='blue', alpha=0.5))
    ax.set_ylabel('WSSE Total', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()


def plot_parameter_histograms(parameters, wsee_ranks, true_params, parameter_names, 
                              stages_to_plot, top_simulations=25, last_stage_only=False):
    """
    Plot KDE histograms for parameters across stages with intensity-based blue colors.
    
    Parameters:
    -----------
    parameters : Dictionary containing parameter arrays for each stage
    wsee_ranks : Dictionary containing WSSE-based ranking indices for each stage
    true_params : True parameter values from field data
    parameter_names : List of parameter names for labeling
    stages_to_plot : List of stage numbers to plot
    top_simulations : Number of top simulations to include based on WSSE ranking
    last_stage_only : If True, only plot the last stage in stages_to_plot
    """
    # Use distinct colors for each stage
    stage_colors = ['black','blue','green','orange','purple','brown','pink','gray','olive']
    
    # Filter stages if last_stage_only is True
    if last_stage_only:
        stages_to_display = [stages_to_plot[-1]]
    else:
        stages_to_display = stages_to_plot
    
    plt.figure(figsize=(20, 15), dpi=150)
    for param_idx in range(15):
        plt.subplot(4, 4, param_idx + 1)
        for stage_idx, stage_num in enumerate(stages_to_display):
            stage_key = f"stage_{stage_num}"
            if stage_key in parameters and stage_key in wsee_ranks:
                # Get top simulations based on WSSE ranking
                top_indices = wsee_ranks[stage_key][:top_simulations]
                param_data = parameters[stage_key][:, param_idx][top_indices]
                
                # Use index from original list for color consistency
                original_idx = stages_to_plot.index(stage_num) if not last_stage_only else len(stages_to_plot) - 1
                color = stage_colors[original_idx % len(stage_colors)]
                sns.kdeplot(param_data, color=color,
                           label=f'Stage {stage_num}', linewidth=2)
        
        # Add true value line
        plt.axvline(true_params[0, param_idx], color='red', linestyle='--', 
                   linewidth=2, label='True Value')
        
        plt.xlabel(parameter_names[param_idx], fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


def calculate_distribution_metrics(parameters, wsee_ranks, true_params, stages_to_analyze, top_simulations=15):
    """
    Calculate metrics to evaluate how well parameter distributions match true values across stages.
    
    Parameters:
    -----------
    parameters : Dictionary containing parameter arrays for each stage
    wsee_ranks : Dictionary containing WSSE-based ranking indices for each stage
    true_params : True parameter values from field data
    stages_to_analyze : List of stage numbers to analyze
    top_simulations : Number of top simulations to include based on WSSE ranking
    Returns:
    --------
    metrics_dict : Dictionary containing various metrics for each stage and parameter
    """
    metrics_results = {}
    for stage_num in stages_to_analyze:
        stage_key = f"stage_{stage_num}"
        if stage_key not in parameters or stage_key not in wsee_ranks:
            continue
        
        stage_metrics = {
            'mean_absolute_error': [],      # MAE between mean and true value
            'bias': [],                      # Mean - true value (signed error)
            'std_dev': [],                   # Standard deviation of distribution
            'coverage_probability': [],      # Does true value fall within 95% CI?
            'distance_to_mode': [],          # Distance from true value to mode (peak)
            'relative_error': [],            # Relative error of mean
            'percentile_rank': [],           # Percentile rank of true value in distribution
        }
        
        top_indices = wsee_ranks[stage_key][:top_simulations]
        
        for param_idx in range(15):
            param_data = parameters[stage_key][:, param_idx][top_indices]
            true_val = true_params[0, param_idx]
            
            # Mean Absolute Error
            mean_val = np.mean(param_data)
            mae = np.abs(mean_val - true_val)
            stage_metrics['mean_absolute_error'].append(mae)
            
            # Bias (signed error)
            bias = mean_val - true_val
            stage_metrics['bias'].append(bias)
            
            # Standard deviation
            std = np.std(param_data)
            stage_metrics['std_dev'].append(std)
            
            # Coverage probability (95% confidence interval)
            ci_lower = np.percentile(param_data, 0)
            ci_upper = np.percentile(param_data, 100)
            coverage = 1.0 if (ci_lower <= true_val <= ci_upper) else 0.0
            stage_metrics['coverage_probability'].append(coverage)
            
            # Distance to mode (using KDE to find mode)
            try:
                kde = stats.gaussian_kde(param_data)
                x_range = np.linspace(param_data.min(), param_data.max(), 200)
                kde_vals = kde(x_range)
                mode_val = x_range[np.argmax(kde_vals)]
                distance_to_mode = np.abs(true_val - mode_val)
                stage_metrics['distance_to_mode'].append(distance_to_mode)
            except:
                stage_metrics['distance_to_mode'].append(np.nan)
            
            # Relative error (normalized by true value)
            if true_val != 0:
                rel_error = np.abs(mean_val - true_val) / np.abs(true_val)
            else:
                rel_error = np.nan
            stage_metrics['relative_error'].append(rel_error)
            
            # Percentile rank of true value
            percentile = stats.percentileofscore(param_data, true_val)
            stage_metrics['percentile_rank'].append(percentile)
        
        for key in stage_metrics:
            stage_metrics[key] = np.array(stage_metrics[key])
        
        metrics_results[stage_key] = stage_metrics
    
    return metrics_results


def summarize_stage_metrics(metrics_results):
    """
    Summarize metrics across all parameters for each stage.    
    Parameters:
    -----------
    metrics_results : Dictionary returned by calculate_distribution_metrics
    Returns:
    --------
    summary : Summary statistics for each stage
    """
    summary = {}
    
    for stage_key, metrics in metrics_results.items():
        summary[stage_key] = {
            'mean_MAE': np.nanmean(metrics['mean_absolute_error']),
            'mean_bias': np.nanmean(metrics['bias']),
            'mean_std': np.nanmean(metrics['std_dev']),
            'coverage_rate': np.nanmean(metrics['coverage_probability']),  # % of params with true value in CI
            'mean_distance_to_mode': np.nanmean(metrics['distance_to_mode']),
            'mean_relative_error': np.nanmean(metrics['relative_error']),
            'median_percentile_rank': np.nanmedian(metrics['percentile_rank']),
        }
    
    return summary


def plot_metric_progression(metrics_results, parameter_names, metric_name='mean_absolute_error', 
                           title=None, ylabel=None):
    """
    Plot how a specific metric evolves across stages for all parameters.
    
    Parameters:
    -----------
    metrics_results : Dictionary returned by calculate_distribution_metrics
    parameter_names : List of parameter names for y-axis labels
    metric_name : Name of the metric to plot
    """
    stage_keys = sorted(metrics_results.keys(), 
                       key=lambda x: int(x.split('_')[1]))
    stage_nums = [int(k.split('_')[1]) for k in stage_keys]
    
    metric_values = []
    for key in stage_keys:
        metric_values.append(metrics_results[key][metric_name])
    
    avg_metric = [np.nanmean(vals) for vals in metric_values]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), dpi=150)
    
    # Plot 1: Average metric across stages
    ax1.plot(stage_nums, avg_metric, marker='o', markersize=10, 
            linewidth=2.5, color='blue', label='Average')
    ax1.set_xlabel('Stage', fontsize=14)
    ax1.set_ylabel(ylabel or metric_name.replace('_', ' ').title(), 
                  fontsize=14)
    ax1.set_title(title or f'{metric_name.replace("_", " ").title()} vs Stage', 
                 fontsize=16)
    ax1.set_xticks(stage_nums)  
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12)
    
    # Plot 2: Heatmap showing metric for each parameter across stages
    metric_array = np.array(metric_values).T  # Shape: (n_params, n_stages)
    im = ax2.imshow(metric_array, aspect='auto', cmap='RdYlGn_r', 
                   interpolation='nearest')
    ax2.set_xlabel('Stage', fontsize=14)
    ax2.set_ylabel('Parameter', fontsize=14)
    ax2.set_xticks(range(len(stage_nums)))
    ax2.set_xticklabels(stage_nums)
    ax2.set_yticks(range(15))
    # Use actual parameter names, truncate if too long
    param_labels = [name[:30] + '...' if len(name) > 30 else name 
                   for name in parameter_names]
    ax2.set_yticklabels(param_labels, fontsize=12)
    plt.colorbar(im, ax=ax2, label=ylabel or metric_name)
    
    plt.tight_layout()
    plt.show()

def compare_conductivity_simulations(true_cond_arranged, simulated_cond_arranged, 
                                    sorted_indices, best_n=3, 
                                    conductivity=True, normalized=False, save_path=None):
    """
    Compare true field conductivity/tracer with simulated data for best samples.
    Shows individual well time series with MSE per well.
    
    Args:
        true_cond_arranged: True field data of shape (8, 3, time_points)
        simulated_cond_arranged: Simulated data of shape (n_samples, 8, 3, time_points)
        sorted_indices: Indices sorted by MSE (best to worst)
        best_n: Number of best simulations to plot
        conductivity: Boolean flag for conductivity (True) or tracer (False)
        normalized: Boolean flag indicating if data is normalized
        save_path: Path to save the figure (optional)
    """
    data_type = 'Conductivity' if conductivity else 'Tracer'
    norm_prefix = 'Normalized ' if normalized else ''
    
    # Define colors for 3 replicates
    rep_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    
    for sample_idx, idx in enumerate(sorted_indices[:best_n]):
        sim_data = simulated_cond_arranged[idx]  # Shape: (8, 3, time_points)
        
        # Calculate R² score
        r2 = r2_score(true_cond_arranged.flatten(), sim_data.flatten())
        
        # Calculate MSE per well
        mse_per_well = np.zeros(8)
        for well in range(8):
            mse_per_well[well] = np.mean((true_cond_arranged[well] - sim_data[well]) ** 2)
        
        # Create figure with 8 well subplots (4x2 grid)
        fig, axes = plt.subplots(4, 2, figsize=(14, 12), dpi=250)
        axes = axes.flatten()
        
        # Plot each well in a separate subplot
        for well in range(8):
            ax = axes[well]
            
            for rep in range(3):
                # Plot conditional data with solid line
                ax.plot(true_cond_arranged[well, rep, :], 
                       color=rep_colors[rep], linestyle='-', linewidth=2, alpha=0.8, 
                       label=f' Conditional W{well+1}_{rep+1}' )
                # Plot inverted data with dashed line (same color)
                ax.plot(sim_data[well, rep, :], 
                       color=rep_colors[rep], linestyle='--', linewidth=2, alpha=0.8, 
                       label=f'Inverted W{well+1}_{rep+1}')
            
            ax.set_xlabel('Time (h)', fontsize=11)
            ax.set_ylabel(f'{norm_prefix}{data_type}', fontsize=11)
            ax.set_title(f'Well {well+1} - MSE: {mse_per_well[well]:.2e}', 
                        fontsize=12)
            ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
            ax.grid(alpha=0.3)
        
        # Overall title
        fig.suptitle(f'WSSE Rank  {sample_idx+1} - {data_type} Comparison', 
                    fontsize=16, y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_path:
            save_path_sample = save_path.replace('.png', f'_sample_{idx}.png')
            plt.savefig(save_path_sample, dpi=250, bbox_inches='tight')
            print(f"Saved plot to: {save_path_sample}")
        plt.show(block=False)

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


def top_ert_simulation_uncertainty(inv_ertd_data, field_ert_true, sorted_indices, wsse_metric_func, save_path=None):
    """
    Plot best simulation ERT image, scatter with ensemble uncertainty band, and KDE envelope.
    Args:
        inv_ertd_data: Inverted ERT data (n_realizations, n_measurements, n_surveys)
        field_ert_true: Field ERT true data
        sorted_indices: Indices sorted by WSSE (best first)
        wsse_metric_func: WSSE metric function
        save_path: Path to save the figure (optional)
    """
    top_simulation = sorted_indices[0]
    WSSE_top, _ = wsse_metric_func(inv_ertd_data[top_simulation, :, :], field_ert_true)
    n_realizations = inv_ertd_data.shape[0]

    universal_fontsize = 16
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), dpi=350)

    # --- Panel 1: Best simulation ERT image ---
    im = axs[0].imshow(inv_ertd_data[top_simulation, :, :], aspect='auto', origin='lower', cmap='jet')
    axs[0].set_title(f'Best simulation (Sim {top_simulation}): WSSE={WSSE_top:.2f}', fontsize=universal_fontsize)
    axs[0].set_xlabel("ERT Surveys", fontsize=universal_fontsize)
    axs[0].set_ylabel("ERT Measurements", fontsize=universal_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=universal_fontsize)
    cbar = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    cbar.set_label('Transfer resistance [Ω]', fontsize=universal_fontsize)
    cbar.ax.tick_params(labelsize=universal_fontsize)

    # --- Panel 2: Scatter with uncertainty ribbon ---
    field_flat = field_ert_true.flatten()
    sort_idx = np.argsort(field_flat)
    field_sorted = field_flat[sort_idx]

    # Stack all realizations and sort by field value
    all_pred = np.array([inv_ertd_data[r].flatten()[sort_idx] for r in range(n_realizations)])
    pred_p5   = np.percentile(all_pred,  5, axis=0)
    pred_p25  = np.percentile(all_pred, 25, axis=0)
    pred_p75  = np.percentile(all_pred, 75, axis=0)
    pred_p95  = np.percentile(all_pred, 95, axis=0)
    #pred_mean = np.mean(all_pred, axis=0)

    min_val = min(field_flat.min(), all_pred.min())
    max_val = max(field_flat.max(), all_pred.max())

    axs[1].fill_betweenx(field_sorted, pred_p5,  pred_p95,  color='steelblue',      alpha=0.20, label='P5–P95 ensemble')
    #axs[1].fill_betweenx(field_sorted, pred_p25, pred_p75,  color='steelblue', alpha=0.35, label='IQR ensemble (P25–P75)')
    #axs[1].plot(pred_mean, field_sorted, color='steelblue', linewidth=1.2, alpha=0.8, label='Ensemble mean')
    r2 = r2_score(field_flat, inv_ertd_data[top_simulation].flatten())
    axs[1].scatter(inv_ertd_data[top_simulation].flatten(), field_flat,
                   color='steelblue', s=6, zorder=5, label=f'Best sim (R²={r2:.4f})')
    axs[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.2, label='1:1 line')
    axs[1].set_xlabel('Inverted ERT [Ω]', fontsize=universal_fontsize)
    axs[1].set_ylabel('Conditional ERT [Ω]', fontsize=universal_fontsize)
    axs[1].legend(fontsize=12)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.3)
    axs[1].tick_params(axis='both', which='major', labelsize=universal_fontsize)

    # --- Panel 3: KDE with ensemble envelope ---
    all_flat_vals = np.concatenate([inv_ertd_data[r].flatten() for r in range(n_realizations)])
    x_grid = np.linspace(all_flat_vals.min(), all_flat_vals.max(), 300)

    kde_curves = np.array([stats.gaussian_kde(inv_ertd_data[r].flatten())(x_grid)
                            for r in range(n_realizations)])
    kde_p5   = np.percentile(kde_curves,  5, axis=0)
    kde_p25  = np.percentile(kde_curves, 25, axis=0)
    kde_p75  = np.percentile(kde_curves, 75, axis=0)
    kde_p95  = np.percentile(kde_curves, 95, axis=0)
    #kde_mean = np.mean(kde_curves, axis=0)

    axs[2].fill_between(x_grid, kde_p5,  kde_p95,  color='steelblue',      alpha=0.25, label='P5–P95 envelope')
    #axs[2].fill_between(x_grid, kde_p25, kde_p75,  color='steelblue', alpha=0.40, label='IQR envelope (P25–P75)')
    #axs[2].plot(x_grid, kde_mean,  color='steelblue', linewidth=1.5, label='Ensemble mean KDE')
    axs[2].plot(x_grid, kde_curves[top_simulation], color='steelblue', linewidth=2.0, label='Best simulation KDE')
    kde_field = stats.gaussian_kde(field_ert_true.flatten())
    axs[2].plot(x_grid, kde_field(x_grid), color='C1', linestyle='--', linewidth=2.0, label='Conditional ERT')
    axs[2].set_xlabel('Transfer resistance [Ω]', fontsize=universal_fontsize)
    axs[2].set_ylabel('Density', fontsize=universal_fontsize)
    axs[2].legend(fontsize=12)
    axs[2].tick_params(axis='both', which='major', labelsize=universal_fontsize)
    axs[2].grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.4)
    w_distance = wasserstein_distance(inv_ertd_data[top_simulation].flatten(), field_ert_true.flatten())
    axs[2].set_title(f'Wasserstein Distance (best): {w_distance:.4f}', fontsize=universal_fontsize)
    axs[2].set_ylim(0, None)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=350, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    plt.show(block=False)


def top_conductivity_uncertainty(true_cond_arranged, simulated_cond_arranged, 
                                    sorted_indices, conductivity=True, normalized=False, save_path=None):
    """
    Plot best simulation conductivity/tracer time series per well with ensemble uncertainty bands.
    Uncertainty is computed across ALL simulations at each time step for each well/replicate.

    Args:
        true_cond_arranged: True field data of shape (8, 3, time_points)
        simulated_cond_arranged: Simulated data of shape (n_samples, 8, 3, time_points)
        sorted_indices: Indices sorted by WSSE (best first)
        conductivity: True for conductivity, False for tracer
        normalized: Whether data is normalized
        save_path: Optional path to save the figure
    """
    data_type = 'Conductivity' if conductivity else 'Tracer'
    norm_prefix = 'Normalized ' if normalized else ''

    rep_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green per replicate
    top_simulation = sorted_indices[0]
    n_samples = simulated_cond_arranged.shape[0]
    time_axis = np.arange(simulated_cond_arranged.shape[3])

    sim_data = simulated_cond_arranged[top_simulation]  # (8, 3, time_points)

    # MSE per well (best vs. true)
    mse_per_well = np.array([
        np.mean((true_cond_arranged[well] - sim_data[well]) ** 2)
        for well in range(8)
    ])

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), dpi=250)
    axes = axes.flatten()

    for well in range(8):
        ax = axes[well]

        for rep in range(3):
            # Ensemble values at each time step: shape (n_samples, time_points)
            ensemble = simulated_cond_arranged[:, well, rep, :]  # (n_samples, T)

            p5  = np.percentile(ensemble,  5, axis=0)
            p95 = np.percentile(ensemble, 95, axis=0)
            ens_median = np.median(ensemble, axis=0)

            c = rep_colors[rep]
            # Outer band P5-P95
            ax.fill_between(time_axis, p5, p95, color=c, alpha=0.12, linewidth=0, label=f'Ensemble W{well+1}_{rep+1} (P5–P95)')
            # Best simulation
            ax.plot(time_axis, sim_data[well, rep, :],
                    color=c, linestyle='--', linewidth=1.8, alpha=0.9,
                    label=f' Inverted W{well+1}_{rep+1}' )
            # Conditional (true) data
            ax.plot(time_axis, true_cond_arranged[well, rep, :],
                    color=c, linestyle='-', linewidth=2.2, alpha=1.0,
                    label=f'Conditional W{well+1}_{rep+1}')

        ax.set_xlabel('Time step', fontsize=12)
        ax.set_ylabel(f'{norm_prefix}{data_type}', fontsize=12)
        ax.set_title(f'Well {well+1}  |  MSE (best): {mse_per_well[well]:.2e}', fontsize=12)
        ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f'Ensemble uncertainty – {data_type}  Best simulation: {top_simulation}',
        fontsize=14, y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        out = save_path.replace('.png', f'_sample_{top_simulation}.png')
        plt.savefig(out, dpi=250, bbox_inches='tight')
        print(f"Saved plot to: {out}")
    plt.show(block=False)


#Results directory
RESULTS_DIR = "/people/hern856/GIFT_Paralell_CDM_WSSE_explore_E2_long_200/GIFT_CDM_Results"
true_params = np.load(f"{RESULTS_DIR}/field_params_true.npy")

config = Config()
data_handler = DataLoaderHandler(config)

PARAMETER_NAMES = [
    '30 Sand cementation exponent',
    '30 Sand tortuosity constant',
    '30 Sand Surface EC [S/m]',
    '30 Sand Porosity',
    '30 Sand Permeability[$m^2$]',
    '70 Sand cementation exponent',
    '70 Sand tortuosity constant',
    '70 Sand Surface EC [S/m]',
    '70 Sand Porosity',
    '70 Sand Permeability[$m^2$]',
    '16 Sand cementation exponent',
    '16 Sand tortuosity constant',
    '16 Sand Surface EC [S/m]',
    '16 Sand Porosity',
    '16 Sand Permeability[$m^2$]'
]

        
DATA_DIC = "/people/hern856/GIFT_Paralell_CDM_WSSE_explore_E2_long_200/Inverted_field_simulations/"

true_params = np.load(f"/people/hern856/GIFT_Paralell_CDM_WSSE_explore_E2_long_200/GIFT_CDM_Results/field_params_true.npy")
print ("True parameters shape:", true_params.shape)
true_ert = np.load(f"/people/hern856/GIFT_Paralell_CDM_WSSE_explore_E2_long_200/GIFT_CDM_Results/field_ert_true.npy")
print ("True ERT shape:", true_ert.shape)
true_cond = np.load(f"/people/hern856/GIFT_Paralell_CDM_WSSE_explore_E2_long_200/GIFT_CDM_Results/field_conductivity_true.npy")
print ("True Conductivity shape:", true_cond.shape)
true_tracer = np.load(f"/people/hern856/GIFT_Paralell_CDM_WSSE_explore_E2_long_200/GIFT_CDM_Results/field_tracer_true.npy")
print ("True Tracer shape:", true_tracer.shape)

true_cond_arranged = well_data_arrange(true_cond,sample_idx=None)
plot_well_data(true_cond_arranged,conductivity=True,normalized=False, sample_idx=None)

true_tracer_arranged = well_data_arrange(true_tracer,sample_idx=None)
plot_well_data(true_tracer_arranged,conductivity=False,normalized=False, sample_idx=None)


verif_ertd = np.load("/people/hern856/GIFT_Paralell_CDM_WSSE_explore_E2_long_200/Inverted_field_simulations/ertd_0_Test.npy").swapaxes(0,1)
verif_params = np.load("/people/hern856/GIFT_Paralell_CDM_WSSE_explore_E2_long_200/Inverted_field_simulations/params_0_Test.npy")
verif_point = np.load("/people/hern856/GIFT_Paralell_CDM_WSSE_explore_E2_long_200/Inverted_field_simulations/obsd_0_Test.npy")
verif_cond = verif_point[0, :, :]
verif_tracer = verif_point[1, :, :]

#Compare the verification sample with the true field data
mse_ert = np.mean((verif_ertd - true_ert)**2)
mse_cond = np.mean((verif_cond - true_cond)**2)
mse_tracer = np.mean((verif_tracer - true_tracer)**2)
mse_params = np.mean((verif_params - true_params)**2)
print(f"MSE between verification ERT and true ERT: {mse_ert:.6f}")
print(f"MSE between verification Conductivity and true Conductivity: {mse_cond:.6f}")
print(f"MSE between verification Tracer and true Tracer: {mse_tracer:.6f}")
print(f"MSE between verification Parameters and true Parameters: {mse_params:.6f}")

#scatter plot of the verification sample vs true field data for conductivity, and ERT
plt.figure(figsize=(12, 5), dpi=150)
plt.subplot(1, 2, 1)
plt.scatter(verif_ertd.flatten(), true_ert.flatten(), color='blue', s=5, alpha=0.7)
plt.plot([true_ert.min(), true_ert.max()], [true_ert.min(), true_ert.max()], color='red', linestyle='--', linewidth=1)
plt.xlabel('Verification ERT [Ω]', fontsize=12)
plt.ylabel('True ERT [Ω]', fontsize=12)
plt.title('Verification ERT vs True ERT', fontsize=14)
plt.grid(alpha=0.3) 
plt.subplot(1, 2, 2)
plt.scatter(verif_cond.flatten(), true_cond.flatten(), color='green', s=5, alpha=0.7)
plt.plot([true_cond.min(), true_cond.max()], [true_cond.min(), true_cond.max()], color='red', linestyle='--', linewidth=1)
plt.xlabel('Verification Conductivity', fontsize=12)    
plt.ylabel('True Conductivity', fontsize=12)
plt.title('Verification Conductivity vs True Conductivity', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



npz_file_path = '/people/hern856/GIFT_Paralell_Uniform_Dataset/Sobol_simulated_data_300/sobol_augmented_data.npz'
data = np.load(npz_file_path)
param_realizations = data['params']
ertd_data = data['ertd']
conductivity_data = data['conductivity']
conductivity_data_arranged = well_data_arrange(conductivity_data)
mse_conductivity = calculate_mse_well_data(true_cond_arranged, conductivity_data_arranged)

tracer_data = data['tracer']
tracer_data_arranged = well_data_arrange(tracer_data)

data_scalers = data_handler.establish_scalers(param_realizations, ertd_data, tracer_data, conductivity_data)


distances_to_true = []
for i in range(ertd_data.shape[0]):
    dist = np.linalg.norm(ertd_data[i] - true_ert)
    distances_to_true.append(dist)
distances_to_true = np.array(distances_to_true)
true_ert_idx = np.argmin(distances_to_true)
#calculate mse between data_true and true_ert
mse_field = np.mean((ertd_data[true_ert_idx] - true_ert)**2)
print(f"\nMSE between most different initial ERT sample and true ERT: {mse_field:.6f}\n")    

#remove the true ert sample from the initial realizations
param_realizations = np.delete(param_realizations, true_ert_idx, axis=0)
ertd_data = np.delete(ertd_data, true_ert_idx, axis=0)
conductivity_data = np.delete(conductivity_data, true_ert_idx, axis=0)
tracer_data = np.delete(tracer_data, true_ert_idx, axis=0)

ert_norm = data_handler.scale_ert_data(ertd_data)
tracer_norm = data_handler.scale_tracer_data(tracer_data)
conductivity_norm = data_handler.scale_conductivity_data(conductivity_data)
param_norm = data_handler.normalize_parameters(param_realizations)


def load_and_evaluate_predictions(
    prediction_type,
    total_stages,
    data_dic,
    data_handler,
    true_ert,
    true_cond,
    true_tracer,
    true_params,
    param_realizations,
    param_norm,
    ertd_data,
    conductivity_data,
    tracer_data,
    true_cond_arranged,
    mad_threshold=9,
):
    """
    Load simulation predictions from disk and evaluate them against field data.

    Parameters
    ----------
    prediction_type : str
        Either 'WSSE_guided' or 'Active_CDM'. Controls which file pattern is
        used and the default number of simulations per stage (30 vs 10).
    total_stages : list[int]
        Stage numbers to load (e.g. [1, 2, ..., 150]).
    data_dic : str
        Directory path containing the simulation .npy files.
    data_handler : DataLoaderHandler
        DataLoaderHandler instance used for normalisation / scaling.
    true_ert : np.ndarray
        Field ERT data used as the WSSE reference.
    true_cond : np.ndarray
        Field conductivity data.
    true_tracer : np.ndarray
        Field tracer data.
    true_params : np.ndarray
        Field (true) parameter values.
    param_realizations : np.ndarray
        Stage-0 (baseline) parameter samples.
    param_norm : np.ndarray
        Normalised version of param_realizations.
    ertd_data : np.ndarray
        Stage-0 ERT simulations.
    conductivity_data : np.ndarray
        Stage-0 conductivity simulations.
    tracer_data : np.ndarray
        Stage-0 tracer simulations.
    true_cond_arranged : np.ndarray
        well_data_arrange(true_cond) – pre-arranged field conductivity.
    mad_threshold : float, optional
        Number of MADs above the median used to flag outliers (default 9).

    Returns
    -------
    parameters, parameters_norm,
    ert_datasets, ert_datasets_norm,
    cond_datasets, cond_datasets_norm,
    tracer_datasets, tracer_datasets_norm,
    wsse_results, wsee_ranks, mse_conductivity_results
    """
    if prediction_type == 'Active_CDM':
        total_simulations = 10
        file_prefix = 'Active_CDM'
    elif prediction_type == 'WSSE_guided':
        total_simulations = 30
        file_prefix = 'WSSE_guided'
    else:
        raise ValueError(f"prediction_type must be 'WSSE_guided' or 'Active_CDM', got '{prediction_type}'")

    parameters        = {}
    parameters_norm   = {}
    ert_datasets      = {}
    ert_datasets_norm = {}
    cond_datasets     = {}
    cond_datasets_norm = {}
    tracer_datasets   = {}
    tracer_datasets_norm = {}
    missing_files_report = {}

    # ── Load active-learning stages ───────────────────────────────────────────
    for stage in total_stages:
        inv_params      = []
        ert_stage_data  = []
        point_stage_data = []
        missing_files   = []

        for sim in range(total_simulations):
            param_file = f"{data_dic}params_{file_prefix}_{stage}_{sim}_stage_{stage}.npy"
            ert_file   = f"{data_dic}ertd_{file_prefix}_{stage}_{sim}_stage_{stage}.npy"
            point_file = f"{data_dic}obsd_{file_prefix}_{stage}_{sim}_stage_{stage}.npy"

            if os.path.exists(param_file) and os.path.exists(ert_file) and os.path.exists(point_file):
                try:
                    inv_params.append(np.load(param_file))
                    ert_stage_data.append(np.load(ert_file))
                    point_stage_data.append(np.load(point_file))
                except Exception as e:
                    print(f"Error loading sim {sim} in stage {stage}: {e}")
                    missing_files.append(sim)
            else:
                missing_files.append(sim)

        if missing_files:
            missing_files_report[f"stage_{stage}"] = missing_files
            print(f"\nStage {stage}: Missing {len(missing_files)} simulations")

        if len(inv_params) > 0:
            inv_params     = np.array(inv_params)
            ert_stage_data = np.array(ert_stage_data).swapaxes(1, 2)
            point_stage_data = np.array(point_stage_data)

            parameters[f"stage_{stage}"]        = inv_params
            parameters_norm[f"stage_{stage}"]   = data_handler.normalize_parameters(inv_params)
            ert_datasets[f"stage_{stage}"]       = ert_stage_data
            ert_datasets_norm[f"stage_{stage}"]  = data_handler.scale_ert_data(ert_stage_data)

            conduct_data_stage = point_stage_data[:, 0, :, :]
            tracer_data_stage  = point_stage_data[:, 1, :, :]

            cond_datasets[f"stage_{stage}"]       = conduct_data_stage
            cond_datasets_norm[f"stage_{stage}"]  = data_handler.scale_conductivity_data(conduct_data_stage)
            tracer_datasets[f"stage_{stage}"]     = tracer_data_stage
            tracer_datasets_norm[f"stage_{stage}"] = data_handler.scale_tracer_data(tracer_data_stage)

            print(f"Stage {stage}: Successfully loaded {len(inv_params)}/{total_simulations} simulations")
        else:
            print(f"WARNING: Stage {stage} has no valid simulations!")

    # ── Add stage-0 / Field baselines ─────────────────────────────────────────
    parameters['stage_0']       = param_realizations
    parameters['Field']         = true_params.reshape(1, -1)
    parameters_norm['stage_0']  = param_norm
    parameters_norm['Field']    = data_handler.normalize_parameters(true_params)

    ert_datasets['Field']        = true_ert.reshape(1, *true_ert.shape)
    ert_datasets['stage_0']      = ertd_data
    ert_datasets_norm['Field']   = data_handler.scale_ert_data(true_ert.reshape(1, *true_ert.shape))
    ert_datasets_norm['stage_0'] = data_handler.scale_ert_data(ertd_data)

    cond_datasets['stage_0']      = conductivity_data
    cond_datasets['Field']        = true_cond.reshape(1, *true_cond.shape)
    cond_datasets_norm['stage_0'] = data_handler.scale_conductivity_data(conductivity_data)
    cond_datasets_norm['Field']   = data_handler.scale_conductivity_data(true_cond.reshape(1, *true_cond.shape))

    tracer_datasets['stage_0']      = tracer_data
    tracer_datasets['Field']        = true_tracer.reshape(1, *true_tracer.shape)
    tracer_datasets_norm['stage_0'] = data_handler.scale_tracer_data(tracer_data)
    tracer_datasets_norm['Field']   = data_handler.scale_tracer_data(true_tracer.reshape(1, *true_tracer.shape))

    # ── Evaluate stage-0 ──────────────────────────────────────────────────────
    wsse_results          = {}
    wsee_ranks            = {}
    mse_conductivity_results = {}

    _, wsse_total_0, sorted_indices_0 = evaluate_simulations_against_field(ert_datasets['stage_0'], true_ert)
    wsse_results['stage_0']          = wsse_total_0[sorted_indices_0]
    wsee_ranks['stage_0']            = sorted_indices_0
    mse_conductivity_results['stage_0'] = calculate_mse_well_data(
        true_cond_arranged, well_data_arrange(cond_datasets['stage_0'], sample_idx=None))

    # ── MAD-based outlier filtering + WSSE evaluation per stage ───────────────
    print(f"\n[UNCERTAINTY FILTERING] Applying MAD-based filtering to active learning stages...")
    for stage in total_stages:
        print(f"\nProcessing stage_{stage}...")
        _, wsse_total, sorted_indices = evaluate_simulations_against_field(
            ert_datasets[f"stage_{stage}"], true_ert)

        # MAD outlier filtering
        median_act          = np.median(wsse_total)
        mad_act             = np.median(np.abs(wsse_total - median_act))
        outlier_threshold   = median_act + mad_threshold * mad_act
        valid_mask          = wsse_total < outlier_threshold
        num_outliers        = np.sum(~valid_mask)

        if np.sum(valid_mask) == 0:
            print(f"  WARNING: All samples marked as outliers! Keeping best sample only.")
            valid_mask             = np.zeros(len(wsse_total), dtype=bool)
            valid_mask[np.argmin(wsse_total)] = True
            num_outliers           = len(wsse_total) - 1

        if num_outliers > 0:
            print(f"  → Removing {num_outliers} extreme samples to prevent training instability")
            for d in (parameters, parameters_norm, ert_datasets, ert_datasets_norm,
                      cond_datasets, cond_datasets_norm, tracer_datasets, tracer_datasets_norm):
                d[f"stage_{stage}"] = d[f"stage_{stage}"][valid_mask]
            wsse_total     = wsse_total[valid_mask]
            sorted_indices = np.argsort(wsse_total)
            print(f"  → Retained {len(parameters[f'stage_{stage}'])} samples")
        else:
            print(f"  → No outliers detected, using all {len(wsse_total)} samples")

        # Arrange well data after filtering
        for d in (cond_datasets, cond_datasets_norm, tracer_datasets, tracer_datasets_norm):
            d[f"stage_{stage}"] = well_data_arrange(d[f"stage_{stage}"])

        wsse_results[f"stage_{stage}"]          = wsse_total[sorted_indices]
        wsee_ranks[f"stage_{stage}"]            = sorted_indices
        mse_conductivity_results[f"stage_{stage}"] = calculate_mse_well_data(
            true_cond_arranged, cond_datasets[f"stage_{stage}"])

    return (parameters, parameters_norm,
            ert_datasets, ert_datasets_norm,
            cond_datasets, cond_datasets_norm,
            tracer_datasets, tracer_datasets_norm,
            wsse_results, wsee_ranks, mse_conductivity_results)


# ── Configuration ─────────────────────────────────────────────────────────────
ALPHA_SD = 0.1
BETA_SD  = 0.01
learning_stages = 150
total_stages    = list(range(1, learning_stages + 1))

# ── Load WSSE-guided predictions ──────────────────────────────────────────────
print("\n" + "="*80)
print("Loading WSSE-guided predictions...")
print("="*80)
(wsse_parameters, wsse_parameters_norm,
 wsse_ert_datasets, wsse_ert_datasets_norm,
 wsse_cond_datasets, wsse_cond_datasets_norm,
 wsse_tracer_datasets, wsse_tracer_datasets_norm,
 wsse_wsse_results, wsse_wsee_ranks, wsse_mse_conductivity_results) = load_and_evaluate_predictions(
    prediction_type   = 'WSSE_guided',
    total_stages      = total_stages,
    data_dic          = DATA_DIC,
    data_handler      = data_handler,
    true_ert          = true_ert,
    true_cond         = true_cond,
    true_tracer       = true_tracer,
    true_params       = true_params,
    param_realizations = param_realizations,
    param_norm        = param_norm,
    ertd_data         = ertd_data,
    conductivity_data = conductivity_data,
    tracer_data       = tracer_data,
    true_cond_arranged = true_cond_arranged,
)

# ── Load Active CDM predictions ───────────────────────────────────────────────
print("\n" + "="*80)
print("Loading Active CDM predictions...")
print("="*80)
(cdm_parameters, cdm_parameters_norm,
 cdm_ert_datasets, cdm_ert_datasets_norm,
 cdm_cond_datasets, cdm_cond_datasets_norm,
 cdm_tracer_datasets, cdm_tracer_datasets_norm,
 cdm_wsse_results, cdm_wsee_ranks, cdm_mse_conductivity_results) = load_and_evaluate_predictions(
    prediction_type   = 'Active_CDM',
    total_stages      = total_stages,
    data_dic          = DATA_DIC,
    data_handler      = data_handler,
    true_ert          = true_ert,
    true_cond         = true_cond,
    true_tracer       = true_tracer,
    true_params       = true_params,
    param_realizations = param_realizations,
    param_norm        = param_norm,
    ertd_data         = ertd_data,
    conductivity_data = conductivity_data,
    tracer_data       = tracer_data,
    true_cond_arranged = true_cond_arranged,
)
# # Save both results to a single .npz for easy downstream loading
# np.savez_compressed(
#     f"{RESULTS_DIR}/cdm_wsse_vs_guided_comparison.npz",
#     cdm_parameters=cdm_parameters,
#     cdm_parameters_norm=cdm_parameters_norm,
#     cdm_ert_datasets=cdm_ert_datasets,
#     cdm_ert_datasets_norm=cdm_ert_datasets_norm,
#     cdm_cond_datasets=cdm_cond_datasets,
#     cdm_cond_datasets_norm=cdm_cond_datasets_norm,
#     cdm_tracer_datasets=cdm_tracer_datasets,
#     cdm_tracer_datasets_norm=cdm_tracer_datasets_norm,
#     cdm_wsse_results=cdm_wsse_results,
#     cdm_wsee_ranks=cdm_wsee_ranks,
#     cdm_mse_conductivity_results=cdm_mse_conductivity_results,
#     wsse_parameters=wsse_parameters,
#     wsse_parameters_norm=wsse_parameters_norm,
#     wsse_ert_datasets=wsse_ert_datasets,
#     wsse_ert_datasets_norm=wsse_ert_datasets_norm,
#     wsse_cond_datasets=wsse_cond_datasets,
#     wsse_cond_datasets_norm=wsse_cond_datasets_norm,
#     wsse_tracer_datasets=wsse_tracer_datasets,
#     wsse_tracer_datasets_norm=wsse_tracer_datasets_norm,
#     wsse_wsse_results=wsse_wsse_results,
#     wsse_wsee_ranks=wsse_wsee_ranks,
#     wsse_mse_conductivity_results=wsse_mse_conductivity_results,
# )
# print(f"\nSaved combined results to: {RESULTS_DIR}/cdm_wsse_vs_guided_comparison.npz\n")



# ── Primary results alias (used by downstream analysis) ──────────────────────
# Change this to 'WSSE_guided' to switch which set drives the rest of the script
primary_prediction_type = 'WSSE_guided'  # or 'Active_CDM'

if primary_prediction_type == 'Active_CDM':
    parameters, parameters_norm           = cdm_parameters, cdm_parameters_norm
    ert_datasets, ert_datasets_norm       = cdm_ert_datasets, cdm_ert_datasets_norm
    cond_datasets, cond_datasets_norm     = cdm_cond_datasets, cdm_cond_datasets_norm
    tracer_datasets, tracer_datasets_norm = cdm_tracer_datasets, cdm_tracer_datasets_norm
    wsse_results, wsee_ranks, mse_conductivity_results = cdm_wsse_results, cdm_wsee_ranks, cdm_mse_conductivity_results
else:
    parameters, parameters_norm           = wsse_parameters, wsse_parameters_norm
    ert_datasets, ert_datasets_norm       = wsse_ert_datasets, wsse_ert_datasets_norm
    cond_datasets, cond_datasets_norm     = wsse_cond_datasets, wsse_cond_datasets_norm
    tracer_datasets, tracer_datasets_norm = wsse_tracer_datasets, wsse_tracer_datasets_norm
    wsse_results, wsee_ranks, mse_conductivity_results = wsse_wsse_results, wsse_wsee_ranks, wsse_mse_conductivity_results


# ── Comparison plot 1: percentile bands – WSSE-guided vs Active CDM ───────────
band_color_wsse = "#4f0cd7"   # purple  – WSSE-guided
band_color_cdm  = "#d7500c"   # orange  – Active CDM

bands = [
    ( 5, 25, 0.12),
    (25, 50, 0.28),
    (50, 75, 0.28),
    (75, 95, 0.12),
]

results_map = {
    'WSSE-Guided': (wsse_wsse_results, band_color_wsse),
    'Active CDM':  (cdm_wsse_results,  band_color_cdm),
}

fig, ax = plt.subplots(figsize=(12, 7), dpi=350)
for label, (res, color) in results_map.items():
    p_vals = {p: np.array([np.percentile(res[f"stage_{s}"], p) for s in total_stages])
              for p in [5, 25, 50, 75, 95]}
    for lo, hi, alpha in bands:
        ax.fill_between(total_stages, p_vals[lo], p_vals[hi],
                        color=color, alpha=alpha, linewidth=0)
    ax.plot(total_stages, p_vals[50], color=color, linestyle='-',
            linewidth=2.0, label=f'{label} – P50', zorder=5)
    # ax.plot(total_stages, p_vals[5],  color=color, linestyle='--',
    #         linewidth=1.0, alpha=0.6, label=f'{label} – P5')
    # ax.plot(total_stages, p_vals[95], color=color, linestyle='--',
    #         linewidth=1.0, alpha=0.6, label=f'{label} – P95')

ax.set_yscale('log')
ax.set_xlabel('Active learning stage', fontsize=14)
ax.set_ylabel('WSSE', fontsize=14)
ax.set_xticks(np.arange(0, np.max(total_stages) + 10, 10))
ax.set_xlim(total_stages[0], total_stages[-1])
ax.set_title('WSSE progression across active learning stages', fontsize=14)
ax.grid(which='major', alpha=0.5, linestyle='-', linewidth=0.8, color='gray')
ax.grid(which='minor', alpha=0.3, linestyle=':', linewidth=0.5, color='gray')
ax.minorticks_on()
legend_entries = [
    Patch(color='none',                                             label='APS CDM:'),
    Patch(color='none',                                             label='APS Samples:'),
    Patch(facecolor=band_color_cdm, alpha=0.15, edgecolor='none', label='P5–P95'),
    Patch(facecolor=band_color_wsse, alpha=0.15, edgecolor='none', label='P5–P95'),
    Patch(facecolor=band_color_cdm, alpha=0.30, edgecolor='none', label='P25–P75'),
    Patch(facecolor=band_color_wsse, alpha=0.30, edgecolor='none', label='P25–P75'),
    Line2D([0],[0], color=band_color_cdm, linewidth=2.5,          label='P50'),
    Line2D([0],[0], color=band_color_wsse, linewidth=2.5,          label='P50'),
]
ax.legend(handles=legend_entries, ncol=4, loc='upper center',
          bbox_to_anchor=(0.5, -0.14), fontsize=11, frameon=True,
          columnspacing=1.0, handlelength=1.8, handletextpad=0.6)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.show()

# ── Comparison plot 2: fine percentile fan – side by side ────────────────────
percentiles = np.arange(0, 96, 1)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=250, sharey=True)
plot_configs = [
    ('WSSE-Guided', wsse_wsse_results, plt.cm.Blues),
    ('Active CDM',  cdm_wsse_results,  plt.cm.Oranges),
]
for ax, (title_str, res, cmap) in zip(axes, plot_configs):
    color_scale = cmap(np.linspace(0.25, 0.95, len(percentiles)))[::-1]
    for idx in range(len(percentiles) - 1):
        upper_vals = [np.percentile(res[f"stage_{s}"], percentiles[idx])     for s in total_stages]
        lower_vals = [np.percentile(res[f"stage_{s}"], percentiles[idx + 1]) for s in total_stages]
        ax.fill_between(total_stages, lower_vals, upper_vals,
                        color=color_scale[idx], linewidth=0)
    ax.set_yscale('log')
    ax.set_xlabel('Active learning stage', fontsize=12)
    ax.set_ylabel('WSSE', fontsize=12)
    ax.set_xticks(np.arange(0, np.max(total_stages) + 10, 10))
    ax.set_xlim(total_stages[0], total_stages[-1])
    ax.set_title(title_str, fontsize=13, fontweight='bold')
    ax.grid(which='major', alpha=0.5, linestyle='-', linewidth=0.8, color='gray')
    ax.grid(which='minor', alpha=0.3, linestyle=':', linewidth=0.5, color='gray')
    ax.minorticks_on()
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=percentiles.min(), vmax=percentiles.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Percentile', fontsize=11)
    tick_values = np.arange(0, percentiles.max() + 1, 10)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{int(t)}' for t in tick_values])
    cbar.ax.tick_params(labelsize=11)

fig.suptitle('WSSE percentile ranges – WSSE-Guided vs Active CDM', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()



# Compare top 3 best conductivity simulations
compare_conductivity_simulations(
    true_cond_arranged, 
    cond_datasets[f"stage_{learning_stages}"],
    wsee_ranks[f"stage_{learning_stages}"],
    best_n=1,
    conductivity=True,
    normalized=False
)

top_conductivity_uncertainty(
    true_cond_arranged, 
    cond_datasets[f"stage_{learning_stages}"],
    wsee_ranks[f"stage_{learning_stages}"],
    conductivity=True,
    normalized=False
)


#Plot top 3 simulations comparison - CDM Samples
plot_best_simulations_comparison(
    ert_datasets[f"stage_{learning_stages}"], true_ert,
    wsee_ranks[f"stage_{learning_stages}"], wsse_metric,
    best_n=3)

top_ert_simulation_uncertainty(
    ert_datasets[f"stage_{learning_stages}"], true_ert,
    wsee_ranks[f"stage_{learning_stages}"], wsse_metric)


LOG_PARAMETERS = [2, 4, 7, 9, 12, 14]
stage_to_plot = learning_stages
#stage_to_plot = 40
fig_kde = plt.figure(figsize=(15, 22), dpi=350)
legend_handles = None
for param_idx in range(15):
    ax = fig_kde.add_subplot(5, 3, param_idx + 1)
    param_data = parameters[f"stage_{stage_to_plot}"][:, param_idx]    
    h1 = ax.hist(param_data, bins=15, density=True, alpha=0.3, color='C0', label='Histogram')
    sns.kdeplot(param_data, ax=ax, linewidth=2, common_norm=False, color='C0', label='KDE')
    ax.axvline(true_params[0, param_idx], color='red', linestyle='-', 
            linewidth=2.5, label='True Value')
    ax.set_xlabel(PARAMETER_NAMES[param_idx], fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    if param_idx in LOG_PARAMETERS:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    else:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style='plain', axis='x')
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()
fig_kde.legend(legend_handles, legend_labels, loc='lower center', ncol=3,
               fontsize=16, frameon=True, bbox_to_anchor=(0.5, 0.03))
plt.tight_layout(rect=[0, 0.06, 1, 0.98])
plt.subplots_adjust(hspace=0.30)
plt.show()

#Formation Factor
ff_30 = parameters[f"stage_{stage_to_plot}"][:, 3]**(-parameters[f"stage_{stage_to_plot}"][:, 0]) 
ff_70 = parameters[f"stage_{stage_to_plot}"][:, 8]**(-parameters[f"stage_{stage_to_plot}"][:, 5])
ff_16 = parameters[f"stage_{stage_to_plot}"][:, 13]**(-parameters[f"stage_{stage_to_plot}"][:, 10])

ff_30_true = parameters[f"Field"][0,3]**(-parameters[f"Field"][0,0]) 
ff_70_true = parameters[f"Field"][0,8]**(-parameters[f"Field"][0,5])
ff_16_true = parameters[f"Field"][0,13]**(-parameters[f"Field"][0,10])

fig_ff = plt.figure(figsize=(18, 5), dpi=350)
ff_data = [(ff_30, ff_30_true, '30 Sand Formation Factor'), (ff_70, ff_70_true, '70 Sand Formation Factor'), (ff_16, ff_16_true, '16 Sand Formation Factor')]
for idx, (ff_values, ff_true, title) in enumerate(ff_data): 
    ax = fig_ff.add_subplot(1, 3, idx + 1)
    h1 = ax.hist(ff_values, bins=15, density=True, alpha=0.3, color='C0', label='Histogram')
    sns.kdeplot(ff_values, ax=ax, linewidth=2, common_norm=False, color='C0', label='KDE')
    ax.axvline(ff_true, color='red', linestyle='-', linewidth=2.5, label='True Value')
    ax.set_xlabel(title, fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
plt.tight_layout(rect=[0, 0.06, 1, 0.98])
plt.subplots_adjust(hspace=0.30)
plt.show()

#Plot parameter scatterplots across stages
up_lim = 0.5
fig, axes = plt.subplots(5, 3, figsize=(16, 22), dpi=350)
axes = axes.flatten()
all_wsse = []
for stage in total_stages:
    stage_key = f"stage_{stage}"
    if stage_key in wsse_results:
        all_wsse.extend(wsse_results[stage_key])
all_wsse = np.array(all_wsse)
#vmin, vmax = np.percentile(all_wsse, [0, 50])
vmin, vmax = np.min(all_wsse),up_lim
scatter_legend_handles = None
for param_idx in range(15):
    ax = axes[param_idx] 
    stage_nums = []
    stage_medians = []
    for stage in total_stages:
        stage_key = f"stage_{stage}"
        if stage_key in parameters and stage_key in wsse_results:
            param_values = parameters[stage_key][:, param_idx]
            wsse_values = wsse_results[stage_key]
            x_positions = np.random.normal(stage, 0.3, size=len(param_values))
            # Color by WSSE (lower is better - use reversed colormap)
            scatter = ax.scatter(x_positions, param_values, c=wsse_values, 
                               cmap='viridis_r', vmin=vmin, vmax=vmax,
                               alpha=0.5, s=15, edgecolors='none')
            stage_nums.append(stage)
            stage_medians.append(np.median(param_values))
    if len(stage_nums) > 0:
        ax.plot(stage_nums, stage_medians, color='darkblue', linestyle='-', 
                linewidth=2, label='Median Trend', zorder=5, alpha=0.8)
    ax.axhline(true_params[0, param_idx], color='red', linestyle='-', 
               linewidth=2.5, label='True Value', zorder=10)
    ax.set_xlabel('Active Learning Stage', fontsize=17)
    ax.set_ylabel('Parameter value', fontsize=17)
    ax.set_title(PARAMETER_NAMES[param_idx], fontsize=17)
    #ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=14)
    if param_idx in LOG_PARAMETERS:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style='plain', axis='y')
    ax.set_xlim(0, learning_stages + 1)
    if scatter_legend_handles is None:
        scatter_legend_handles, scatter_legend_labels = ax.get_legend_handles_labels()
# Layout first, then overlay legend and colorbar into the reserved margin
plt.tight_layout()
plt.subplots_adjust(bottom=0.10)
fig.legend(scatter_legend_handles, scatter_legend_labels, loc='lower center',
           ncol=2, fontsize=16, frameon=True, bbox_to_anchor=(0.5, 0.045))
cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.018])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('WSSE Value (Lower = Better Fit)', fontsize=18)
cbar.ax.tick_params(labelsize=16)
# Customize colorbar ticks and labels
tick_positions = [vmin, vmin + (vmax - vmin) * 0.25, vmin + (vmax - vmin) * 0.5, 
                  vmin + (vmax - vmin) * 0.75, vmax]
tick_labels = [f'{vmin:.2f}', f'{vmin + (vmax - vmin) * 0.25:.2f}', 
               f'{vmin + (vmax - vmin) * 0.5:.2f}', f'{vmin + (vmax - vmin) * 0.75:.2f}', f'>{up_lim:.2f}']
cbar.set_ticks(tick_positions)
cbar.set_ticklabels(tick_labels)
plt.show()


# Plot parameter distributions with histograms and KDE across stages

# Collect all WSSE values for normalization
all_wsse = []
for stage in total_stages:
    stage_key = f"stage_{stage}"
    if stage_key in wsse_results:
        all_wsse.extend(wsse_results[stage_key])
all_wsse = np.array(all_wsse)
vmin, vmax = np.min(all_wsse), up_lim
# Plot each formation factor
ff_data = [
    (3, 0, ff_30_true, '30 Sand Formation Factor'),
    (8, 5, ff_70_true, '70 Sand Formation Factor'),
    (13, 10, ff_16_true, '16 Sand Formation Factor')]

#Formation Factor Evolution as Scatter Plot colored by WSSE
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=350)
for col_idx, (param_idx1, param_idx2, ff_true, title) in enumerate(ff_data):
    ax = axes[col_idx]
    stage_medians = []
    stage_nums_for_line = []
    
    for stage in total_stages:
        stage_key = f"stage_{stage}"
        if stage_key in parameters and stage_key in wsse_results:
            ff_values = parameters[stage_key][:, param_idx1]**(-parameters[stage_key][:, param_idx2])
            wsse_values = np.array(wsse_results[stage_key])
            x_positions = np.random.normal(stage, 0.2, size=len(ff_values))
            scatter = ax.scatter(x_positions, ff_values, c=wsse_values, 
                               cmap='viridis_r', vmin=vmin, vmax=vmax,
                               alpha=0.6, s=30, edgecolors='none')
            
            stage_medians.append(np.median(ff_values))
            stage_nums_for_line.append(stage)
    if len(stage_nums_for_line) > 0:
        ax.plot(stage_nums_for_line, stage_medians, color='darkblue', linestyle='-', 
                linewidth=2, label='Median Trend', zorder=5, alpha=0.8)
    ax.axhline(ff_true, color='red', linestyle='-', 
               linewidth=2.5, label='True Value', zorder=10)
    ax.set_xlabel('Learning Stage', fontsize=17)
    #ax.set_ylabel('Formation Factor', fontsize=17)
    ax.set_ylabel(title, fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='y')
    ax.set_xlim(0, 150)
    ax.set_xticks(np.arange(0, 151, 25))
    ax.legend(fontsize=16)
# Layout first, then overlay legend and colorbar into the reserved margin
plt.tight_layout()
plt.subplots_adjust(bottom=0.20)
cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.05])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('WSSE Value (Lower = Better Fit)', fontsize=18)
cbar.ax.tick_params(labelsize=16)
# Customize colorbar ticks and labels
tick_positions = [vmin, vmin + (vmax - vmin) * 0.25, vmin + (vmax - vmin) * 0.5, 
                  vmin + (vmax - vmin) * 0.75, vmax]
tick_labels = [f'{vmin:.2f}', f'{vmin + (vmax - vmin) * 0.25:.2f}', 
               f'{vmin + (vmax - vmin) * 0.5:.2f}', f'{vmin + (vmax - vmin) * 0.75:.2f}', f'>{up_lim:.2f}']
cbar.set_ticks(tick_positions)
cbar.set_ticklabels(tick_labels)
plt.show()




stages = [1]+[i for i in np.arange(5, stage_to_plot, 5)]+[stage_to_plot]
fig, axes = plt.subplots(3, 5, figsize=(28, 15), dpi=300)
axes = axes.flatten()
palette = sns.color_palette("inferno_r", len(stages))
for i in range(15):
    if i < len(axes):
        ax = axes[i]        
        for idx, stage in enumerate(stages):
            stage_key = f"stage_{stage}"
            if stage_key in parameters:
                param_data = parameters[stage_key][:, i]
                if len(param_data) > 1:
                    # Plot histogram with transparency
                    ax.hist(param_data, bins=20, density=True, alpha=0.15, 
                           color=palette[idx], edgecolor=palette[idx], linewidth=0.5)
                    # Plot KDE on top
                    sns.kdeplot(param_data, ax=ax, color=palette[idx], 
                               fill=False, linewidth=2, alpha=0.8)        
        # Plot true value
        ax.axvline(true_params[0][i], color='red', linestyle='--', 
                  linewidth=2.5, label='True Value', zorder=10)
        
        ax.set_xlabel(f"{PARAMETER_NAMES[i]}", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cmap = ListedColormap(palette)
bounds = np.arange(len(stages) + 1)
norm = BoundaryNorm(bounds, cmap.N)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                  cax=cbar_ax, ticks=np.arange(len(stages)) + 0.5)
cb.set_label('Active learning stage', fontsize=14, rotation=270, labelpad=20)
cb.ax.set_yticklabels([f'{stage}' for stage in stages], fontsize=12)
plt.suptitle('Parameter Distributions with Histograms & KDE Across Active Learning Stages', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 0.91, 0.99])
plt.show()



mds_stages_to_include = [0]+stages
plot_mds_projection(parameters, stages_to_include=mds_stages_to_include, title="MDS Projection - CDM Samples", 
                   print_diagnostics=False)

true_ert_norm = data_handler.scale_ert_data(true_ert.reshape(true_ert.shape))
ert_mds_stages_to_include = [0] + stages  # Include stage_0 and active learning stages
plot_mds_projection_ert(ert_datasets_norm, true_ert_norm, 
                       stages_to_include=ert_mds_stages_to_include, 
                       title="MDS Projection - ERT Predictions", 
                       show_zoom=True)

plot_simulations_comparisons(
    ert_datasets[f"stage_{learning_stages}"], true_ert,
    wsee_ranks[f"stage_{learning_stages}"], wsse_metric)


# Plot correlation matrices for each stage
for stage in mds_stages_to_include:
    stage_key = f"stage_{stage}"
    if stage_key in parameters:
        # Get parameter data for this stage
        stage_params = parameters[stage_key]
        corr_matrix = np.corrcoef(stage_params.T)
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f'Pearson Correlation - Stage: {stage}', fontsize=12, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=10)
        ax.set_xticks(np.arange(15))
        ax.set_yticks(np.arange(15))
        param_labels = [f'P{i+1}' for i in range(15)]
        ax.set_xticklabels(param_labels, fontsize=10, rotation=45, ha='right')
        ax.set_yticklabels(param_labels, fontsize=10)
        for i in range(15):
            for j in range(15):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                             fontsize=7)
        ax.axhline(y=4.5, color='black', linewidth=2, linestyle='-')
        ax.axhline(y=9.5, color='black', linewidth=2, linestyle='-')
        ax.axvline(x=4.5, color='black', linewidth=2, linestyle='-')
        ax.axvline(x=9.5, color='black', linewidth=2, linestyle='-')
        ax.text(-1.8, 2, 'Sand 30', fontsize=11, fontweight='bold', rotation=90, 
                verticalalignment='center', horizontalalignment='center')
        ax.text(-1.8, 7, 'Sand 70', fontsize=11, fontweight='bold', rotation=90, 
                verticalalignment='center', horizontalalignment='center')
        ax.text(-1.8, 12, 'Sand 16', fontsize=11, fontweight='bold', rotation=90, 
                verticalalignment='center', horizontalalignment='center')
        ax.text(2, -1.0, 'Sand 30', fontsize=11, fontweight='bold', 
                verticalalignment='center', horizontalalignment='center')
        ax.text(7, -1.0, 'Sand 70', fontsize=11, fontweight='bold', 
                verticalalignment='center', horizontalalignment='center')
        ax.text(12, -1.0, 'Sand 16', fontsize=11, fontweight='bold', 
                verticalalignment='center', horizontalalignment='center')
        legend_text = "Parameter Index Legend:\n" + "\n".join([f"P{i+1}: {PARAMETER_NAMES[i]}" for i in range(15)])
        fig.text(0.78, 0.5, legend_text, fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='white', linewidth=1))
        plt.tight_layout(rect=[0, 0, 0.77, 1])  # Leave space for legend on the right
        plt.show()

#Plot a histogram of the best simulation parameters below WSSE of 2 across all stages
best_wsse_threshold = 1
best_simulation_params = []
for stage in total_stages:
    stage_key = f"stage_{stage}"
    if stage_key in parameters and stage_key in wsee_ranks:
        stage_params = parameters[stage_key]
        stage_wsse = wsse_results[stage_key]
        best_indices = np.where(stage_wsse < best_wsse_threshold)[0]
        best_simulation_params.append(stage_params[best_indices])
best_simulation_params = np.vstack(best_simulation_params)
print (f"\nTotal simulations with WSSE < {best_wsse_threshold}: {len(best_simulation_params)}")

plt.figure(figsize=(12, 8), dpi=300)
for i in range(15):
    plt.subplot(4, 4, i + 1)
    param_data = best_simulation_params[:, i]
    plt.hist(param_data, bins=30, density=True, alpha=0.3, color='steelblue', edgecolor='black')
    sns.kdeplot(param_data, linewidth=2, color='navy')
    plt.axvline(true_params[0, i], color='red', linestyle='--', linewidth=2, label='True Value')
    plt.xlabel(PARAMETER_NAMES[i], fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
plt.tight_layout()
plt.suptitle(f'Parameter Distributions for Simulations with WSSE < {best_wsse_threshold}', 
                fontsize=16, fontweight='bold', y=0.995)
plt.show()


#%% Calculate and visualize distributions metrics

metrics_results = calculate_distribution_metrics(
    parameters, wsee_ranks, true_params, total_stages, top_simulations=30)

summary = summarize_stage_metrics(metrics_results)


# Print summary - CDM Samples
print("\n" + "="*80)
print("DISTRIBUTION METRICS SUMMARY - CDM SAMPLES")
print("="*80)
for stage_key in sorted(summary.keys(), key=lambda x: int(x.split('_')[1])):
    stage_num = stage_key.split('_')[1]
    print(f"\nStage {stage_num}:")
    print(f"  Mean Absolute Error:     {summary[stage_key]['mean_MAE']:.6f}")
    print(f"  Mean Bias:               {summary[stage_key]['mean_bias']:.6f}")
    print(f"  Mean Std Dev:            {summary[stage_key]['mean_std']:.6f}")
    print(f"  Coverage Rate (95% CI):  {summary[stage_key]['coverage_rate']*100:.1f}%")
    print(f"  Distance to Mode:        {summary[stage_key]['mean_distance_to_mode']:.6f}")
    print(f"  Mean Relative Error:     {summary[stage_key]['mean_relative_error']:.6f}")
    print(f"  Median Percentile Rank:  {summary[stage_key]['median_percentile_rank']:.1f}")

# Plot metric progressions - CDM Samples
plot_metric_progression(metrics_results, PARAMETER_NAMES, 'mean_absolute_error',
                       title='Mean Absolute Error vs Stage - CDM Samples',
                       ylabel='MAE')

plot_metric_progression(metrics_results, PARAMETER_NAMES, 'coverage_probability',
                       title='Coverage Probability (95% CI) vs Stage - CDM Samples',
                       ylabel='Coverage (1=true value in CI)')

plot_metric_progression(metrics_results, PARAMETER_NAMES, 'std_dev',
                       title='Standard Deviation vs Stage - CDM Samples',
                       ylabel='Standard Deviation')

plot_metric_progression(metrics_results, PARAMETER_NAMES, 'relative_error',
                       title='Relative Error vs Stage - CDM Samples',
                       ylabel='Relative Error')

plot_metric_progression(metrics_results, PARAMETER_NAMES, 'bias',
                       title='Bias vs Stage - CDM Samples',
                       ylabel='Bias')

plot_metric_progression(metrics_results, PARAMETER_NAMES, 'distance_to_mode',
                       title='Distance to Mode vs Stage - CDM Samples',
                       ylabel='Distance to Mode')


