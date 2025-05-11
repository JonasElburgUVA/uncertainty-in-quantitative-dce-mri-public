from enum import Enum

class NetworkType(str, Enum):
    FC = "fc"
    FC_UNC = "fc_unc"
    DCENET = "dcenet"
    DCENET_UNC = "dcenet_unc"

class ModelType(str, Enum):
    SNN = "snn"
    PINN = "pinn"
    MVE_SNN = "mve_snn"
    MVE_PINN = "mve_pinn"
    PINN_PH = "pinn_ph"

class DataType(str, Enum):
    NORMAL = "normal"
    # INVIVO = "invivo"
    OOD = "ood"
    VIVO = "vivo"
