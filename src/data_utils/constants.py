from enum import Enum

# CheXpert and MIMIC-CXR labels
CXR_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CXR_SHORT_LABELS = [
    "NF",
    "EC",
    "Cm",
    "LO",
    "LL",
    "Ed",
    "Co",
    "Pn",
    "At",
    "Px",
    "PE",
    "PO",
    "Fr",
    "SD",
]
CXP_CHALLENGE_LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

CXP_CHALLENGE_LABELS_IDX = [CXR_LABELS.index(l) for l in CXP_CHALLENGE_LABELS]


class CXRLabelStrategy(Enum):
    UNCERTAIN_ZEROS = 0
    UNCERTAIN_ONES = 1
    UNCERTAIN_RANDOM = 2


# Fitzpatrick17k labels
FITZPATRICK_LABELS_COARSE = ["non-neoplastic", "malignant", "benign"]
FITZPATRICK_LABELS_COARSE_SHORT = ["NN", "M", "B"]
FITZPATRICK_LABELS_FINE = [
    "inflammatory",
    "malignant epidermal",
    "genodermatoses",
    "benign dermal",
    "benign epidermal",
    "malignant melanoma",
    "benign melanocyte",
    "malignant cutaneous lymphoma",
    "malignant dermal",
]
FITZPATRICK_LABELS_FINE_SHORT = [
    "In",
    "ME",
    "Ge",
    "BD",
    "BE",
    "MM",
    "BM",
    "MCL",
    "MD",
]
EMBED_LABELS = [
    "BIRADS-A",
    "BIRADS-B",
    "BIRADS-C",
    "BIRADS-D",
]
EMBED_LABELS_SHORT = ["A", "B", "C", "D"]
EMBED_EXAM_LABELS = ["A", "N", "B", "P", "S", "M", "K", "X"]
EMBED_EXAM_LABELS_BIRADS = [0, 1, 2, 3, 4, 5, 6]
FAIRVISION_LABELS = [
    "healthy",
    "non-vision threatening dr",
    "vision threatening dr",
    "glaucoma",
    "early amd",
    "intermediate amd",
    "late amd",
]
FAIRVISION_LABELS_SHORT = ["H", "NVT-DR", "VT-DR", "G", "E-AMD", "I-AMD", "L-AMD"]

ECG_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
ECG_LABELS_LONG = ["Normal", "myocardial infarction", "ST/T-changes", "conduction disturbance", "hypertrophy"]