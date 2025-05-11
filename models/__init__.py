from .snn import SNN
from .pinn import PINN
from .mve_snn import MVE_SNN
from .mve_pinn import MVE_PINN
from .pinn_ph import PINN_PH

from utils.enums import ModelType

from .ensemble import EnsembleWrapper

def get_model(model_name, network,
              weights=None,
              mode="sim",
              loggertrue=False,
              lr=1e-4,
              **kwargs):
    '''
    Returns the model class based on the model_name
    '''
    model = {
        ModelType.SNN : SNN,
        ModelType.PINN : PINN,
        ModelType.MVE_SNN : MVE_SNN,
        ModelType.MVE_PINN : MVE_PINN, 
        ModelType.PINN_PH : PINN_PH
    }[model_name](network=network, weights=weights, mode=mode,lr=lr, loggertrue=loggertrue, **kwargs)
    
    return model

def get_ensemble(models, **kwargs):
    '''
    Returns a list of n_models networks
    '''
    ensemble = EnsembleWrapper(models, **kwargs)

    return ensemble

