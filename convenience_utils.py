from pathlib import Path
from src.colors import (chex_color, embed_color, fairvision_color,
                              fitz_color, mimic_color, ptb_xl_color, mimic_iv_ed_color)

def fig_dir_exists(out_dir: Path):
    """
    Ensure output directory structure exists for saving figures and files.
    
    Creates the main output directory and a 'files' subdirectory if they don't exist.
    Used for organizing plot outputs and exported data files.
    
    Args:
        out_dir: Path to the main output directory
    """
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    files_dir = out_dir / "files"
    if not files_dir.exists():
        files_dir.mkdir(parents=True, exist_ok=True)


def get_patient_col(dataset_name: str):
    """
    Get the patient ID column name for a given medical dataset.
    
    Different medical datasets use different column names to identify patients.
    This function provides a standardized way to retrieve the correct column name
    for patient-level aggregation and privacy analysis.
    
    Args:
        dataset_name: Short name of the dataset
        
    Returns:
        str: Column name used for patient identification
        
    Raises:
        ValueError: If dataset name is not recognized
        
    Examples:
        >>> get_patient_col("chexpert")
        "patient_id"
        >>> get_patient_col("mimic")
        "subject_id"
    """
    if dataset_name == "chexpert" or dataset_name == "fairvision":
        patient_col = "patient_id"
    elif dataset_name == "mimic":
        patient_col = "subject_id"
    elif dataset_name == "embed":
        patient_col = "empi_anon"
    elif dataset_name == "fitzpatrick":
        patient_col = "md5hash" # we assume each image belongs to a unique patient
    elif dataset_name == "ptb-xl":
        patient_col = "patient_id"
    elif dataset_name == "mimic-iv-ed":
        patient_col = "subject_id"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return patient_col


def get_data_root(dataset_name: str):
    """
    Get the filesystem path to the root directory of a medical dataset.
    
    Returns the absolute path where dataset files are stored on the local filesystem.
    These paths are environment-specific and may need to be adjusted for different
    computing environments.
    
    Args:
        dataset_name: Short name of the dataset
        
    Returns:
        Path or None: Absolute path to dataset root directory, or None if dataset
                     doesn't use local files (e.g., MIMIC-IV-ED uses database)
        
    Raises:
        ValueError: If dataset name is not recognized
        
    Note:
        Paths are configured for the original research environment and may need
        adjustment for different setups.
    """
    if dataset_name == "chexpert":
        data_root = Path("/home/moritz/data/chexpert")
    elif dataset_name == "mimic":
        data_root = Path("/home/moritz/data/mimic-cxr/mimic-cxr-jpg")
    elif dataset_name == "fairvision":
        data_root = Path("/home/moritz/data_big/fairvision/FairVision")
    elif dataset_name == "embed":
        data_root = Path("/home/moritz/data_massive/embed_small/png/1024x768")
    elif dataset_name == "fitzpatrick":
        data_root = Path("/home/moritz/data/fitzpatrick17k")
    elif dataset_name == "ptb-xl":
        data_root = Path("/home/moritz/data/physionet.org/files/ptb-xl/1.0.3/")
    elif dataset_name == "mimic-iv-ed":
        data_root = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data_root

def get_color(dataset_name):
    """
    Get the standardized color for visualizing a specific medical dataset.
    
    Returns a consistent color scheme used across all plots and visualizations
    to maintain visual consistency when comparing results across datasets.
    
    Args:
        dataset_name: Short name of the dataset
        
    Returns:
        Color value (format depends on src.colors implementation)
        
    Raises:
        ValueError: If dataset name is not recognized
        
    Note:
        Colors are defined in src.colors module and designed for accessibility
        and visual distinction across different medical datasets.
    """
    if dataset_name == "chexpert":
        return chex_color
    elif dataset_name == "mimic":
        return mimic_color
    elif dataset_name == "fitzpatrick":
        return fitz_color
    elif dataset_name == "fairvision":
        return fairvision_color
    elif dataset_name == "embed":
        return embed_color
    elif dataset_name == "ptb-xl":
        return ptb_xl_color
    elif dataset_name == "mimic-iv-ed":
        return mimic_iv_ed_color
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}.")



def get_dataset_name(dataset_name):
    """
    Convert dataset short name to human-readable display name.
    
    Transforms internal dataset identifiers to properly formatted names
    suitable for academic publications, plots, and user interfaces.
    
    Args:
        dataset_name: Short internal identifier for the dataset
        
    Returns:
        str: Formatted display name for publications and visualizations
        
    Raises:
        ValueError: If dataset name is not recognized
        
    Examples:
        >>> get_dataset_name("chexpert")
        "CheXpert"
        >>> get_dataset_name("mimic")
        "MIMIC-CXR"
        >>> get_dataset_name("fitzpatrick")
        "Fitzpatrick 17k"
    """
    if dataset_name == "chexpert":
        return "CheXpert"
    elif dataset_name == "mimic":
        return "MIMIC-CXR"
    elif dataset_name == "fitzpatrick":
        return "Fitzpatrick 17k"
    elif dataset_name == "fairvision":
        return "FairVision"
    elif dataset_name == "embed":
        return "EMBED"
    elif dataset_name == "ptb-xl":
        return "PTB-XL"
    elif dataset_name == "mimic-iv-ed":
        return "MIMIC-IV ED"
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}.")

