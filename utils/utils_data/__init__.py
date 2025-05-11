from ..enums import DataType
from .datamodule import MyDataModule


def get_dataset(
        dataset_name,
        batch_size=32,
):
    '''
    Returns the dataset class based on the dataset_name
    '''
    path = 'data'
    kwargs = {
        DataType.NORMAL : {
            'datapath': path,
            'mode': 'sim',
            'simmode': 'normal',
            'SNR': True
        },
        DataType.OOD : {
            'datapath': path,
            'mode': 'sim',
            'simmode': 'normal',
            'OOD': True,
            'SNR': True
        },
        DataType.VIVO : {
            'datapath': path,
            'mode': 'vivo',
            'SNR': False
        }
    }[dataset_name]

    dataset = MyDataModule(batch_size=batch_size, **kwargs)
    return dataset
