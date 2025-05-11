from networks.mlp import FC
from networks.mlp import FC_UNC
from networks.dcenet import DCENET, DCENET_UNC

from utils.enums import NetworkType
import torch

def get_network(network_name, **kwargs):
    '''
    Returns the network class based on the network_name
    '''
    network = {
        NetworkType.FC : FC,
        NetworkType.FC_UNC : FC_UNC,
        NetworkType.DCENET : DCENET,
        NetworkType.DCENET_UNC : DCENET_UNC,
    }[network_name](**kwargs)
    
    return network


