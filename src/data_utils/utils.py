

def get_dataset_str(dataset: str):
    """
    Convert dataset short name to human-readable display name.
    
    Args:
        dataset: Short dataset identifier (e.g., 'mimic', 'chexpert')
        
    Returns:
        str: Human-readable dataset name for plots and displays
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    if dataset == "mimic":
        return "MIMIC-CXR"
    elif dataset == "chexpert":
        return "CheXpert"
    elif dataset == "fitzpatrick":
        return "Fitzpatrick 17k"
    elif dataset == "ukbb_cfp":
        return "UKBB-CFP"
    elif dataset == "embed":
        return "EMBED"
    elif dataset == "fairvision":
        return "FairVision"
    elif dataset == "ptb-xl":
        return "PTB-XL"
    elif dataset == "mimic-iv-ed":
        return "MIMIC-IV-ED"
    raise ValueError(f"Invalid dataset name., {dataset}")
