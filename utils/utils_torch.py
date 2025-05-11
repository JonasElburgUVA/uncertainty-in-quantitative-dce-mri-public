"""
This file contains the torch implementation of the Extended Tofts model and helper functions to convert between concentration and parameters, necessary for deep fitting methods.
"""
import torch
from torch.nn.functional import sigmoid
import tqdm
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import numpy as np


"""
GLOBAL VARIABLES
"""
if torch.cuda.is_available():
    Device = torch.device("cuda")
elif torch.backends.mps.is_available():
    Device = torch.device("mps")
else:
    Device = torch.device("cpu")
Ranges = torch.tensor([3., 2., 1., 1.]).to(Device)
Bounds = torch.tensor([[0., 0., 0., 0.], [3., 2., 1., 1.]]).to(Device)


"""
FUNCTIONS
"""
def sigmoid_normalise(x):
    return sigmoid(x) * Ranges * 1.5 - 0.25 * Ranges

"""
TOFTS FUNCTIONS
"""
def aifPopPMB(Hct = 0.4):
    # defines plasma curve; note this is a population-based AIF based on https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21066 and https://iopscience.iop.org/article/10.1088/0031-9155/53/5/005
    # NB the PMB paper neglected to include the Hct in the parameter ab, so we include this adjustment here with an assumed Hct
    aif = {'ab': 2.84/(1-Hct), 'mb': 22.8, 'ae': 1.36, 'me': 0.171, 't0': 0.2}
    return aif


def Cosine4AIF_ExtKety(t, aif, ke, dt, ve, vp):
    '''
    expects torch tensors, with constants:
    t: timesteps [T]
    aif: arterial input function as a dict of 
        {
            't0' : t0 [1],
            'ab' : ab [1],
            'mb' : mb [1],
            'me' : me [1],
            'ae' : ae [1]
        }

    and trainable parameters:
    ke: ke [B]
    dt: dt [B]
    ve: ve [B]
    vp: vp [B]
    '''
    device = ke.device
    t.to(device)
    # create timesteps for each sample in the batch
    B = ke.shape[0]
    T = t.shape[0]
    t = t.unsqueeze(0).repeat(B, 1) # [B, T]
    dt = dt.unsqueeze(1).repeat(1, T) # [B, T]
    ke = ke.unsqueeze(1).repeat(1, T) # [B, T]
    vp = vp.unsqueeze(1).repeat(1, T) # [B, T]
    ve = ve.unsqueeze(1).repeat(1, T) # [B, T]

    for k,v in aif.items():
        aif[k] = torch.tensor(v).unsqueeze(0).unsqueeze(0).repeat(B, T).to(device)

    try:
        t = t - aif['t0'] - dt #potentially need to unsqueeze dt; dt has to subtracted from each column in t
    except:
        assert 1==0

    cpBolus = aif['ab'] * CosineBolus(t,aif['mb']) # [B, T]
    cpWashout = aif['ab'] * aif['ae'] * ConvBolusExp(t, aif['mb'], aif['me']) # [B, T]    
    ceBolus = ke * aif['ab'] * ConvBolusExp(t, aif['mb'], ke) # [B,T]
    ceWashout = ke * aif['ab'] * aif['ae'] * ConvBolusExpExp(t, aif['mb'], aif['me'], ke) # [B,T]

    cp = cpBolus + cpWashout
    ce = ceBolus + ceWashout

    ct = torch.zeros_like(t, dtype=t.dtype) # [B, T]
    mf = (t>=0).to(dtype=t.dtype)
    # ct[t>=0] = ((mf * vp * cp) + (mf * ve * ce))[t>=0]
    ct = torch.where(t>=0, (mf * vp * cp) + (mf * ve * ce), ct)

    return ct

def etofts(x, batching=False):
    device = x.device
    b = x.shape[0]
    t = (torch.arange(0, 80, 1) * 4 / 60).to(device)

    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    # dimensions give the 4 parameters of the etofts model: ke, dt, ve, vp. Note, aif = Cp
    ke = x[:, 0]
    dt = x[:, 1]
    ve = x[:, 2]
    vp = x[:, 3]

    if batching:
        bs = 1024
        ct = torch.empty((b, 80), device=device)
        for i in tqdm.tqdm(range(0, b, bs)):
            aif = aifPopPMB()
            end = min(i+bs, b)
            ct[i:end] = Cosine4AIF_ExtKety(t, aif, ke[i:end], dt[i:end], ve[i:end], vp[i:end])

    else:
        aif = aifPopPMB()

        ct = Cosine4AIF_ExtKety(t, aif, ke, dt, ve, vp)
    return ct

def ConvBolusExpExp(t, m, k1, k2):
    tol = 1e-4

    tT = tol / torch.abs(k1-k2)
    Ig = (t > 0) & (t < tT) # shapes...
    Igf = Ig.to(dtype=torch.int32)
    Ie = t >= tT
    Ief = Ie.to(dtype=torch.int32)
    y = torch.zeros_like(t)
    y1 = torch.zeros_like(t)
    y2 = torch.zeros_like(t)

    # y[Ig] = ConvBolusGamma(t, m, 0.5*(k1+k2))[Ig]
    y = torch.where(Ig, ConvBolusGamma(t, m, 0.5*(k1+k2)), y)
    # y1 = ConvBolusExp(t*Ief, m, k1)[Ie]
    y1 = torch.where(Ie, ConvBolusExp(t*Ief, m, k1), y1)
    # y2 = ConvBolusExp(t*Ief, m, k2)[Ie]
    y2 = torch.where(Ie, ConvBolusExp(t*Ief, m, k2), y2)
    try: 
        # y[Ie] = (y1 - y2) / (k2[Ie] - k1[Ie])
        # y = torch.where(Ie, (y1 - y2) / (k2[Ie] - k1[Ie]), y)
        y = torch.where(Ie, (y1 - y2) / (k2 - k1), y)
    except IndexError:
        # y[Ie] = (y1 - y2) / (k2 - k1)
        y = torch.where(Ie, (y1 - y2) / (k2 - k1), y)

    return y

def ConvBolusGamma(t, m, k):
    tB = 2 * torch.pi / m
    y = torch.zeros_like(t)
    I1 = (t >= 0) & (t <= tB)
    I2 = t > tB
    I1f = I1.to(dtype=torch.int32)
    I2f = I2.to(dtype=torch.int32)

    ce = SpecialCosineExp(k * tB, m * tB)
    cg = SpecialCosineGamma(k * tB, m * tB)
    # y[I1] = ((t*I1f)**2 * SpecialCosineGamma(k * t*I1f, m* t*I1f))[I1]
    # y[I2] = (tB * (((t*I2f - tB) * ce + tB * cg) * torch.exp(-k * (t*I2f - tB))))[I2]

    y = torch.where(I1, (t*I1f)**2 * SpecialCosineGamma(k * t*I1f, m* t*I1f), y)
    y = torch.where(I2, tB * (((t*I2f - tB) * ce + tB * cg) * torch.exp(-k * (t*I2f - tB))), y)

    return y

def SpecialCosineGamma(x,y):
    epsilon = 1e-8
    expTerm = (3 + (y ** 2) / (x ** 2 + epsilon)) * (1 - torch.exp(-x)) \
          - ((y ** 2 + x ** 2) / (x + epsilon)) * torch.exp(-x)
    trigTerm = ((x ** 2 - y ** 2) * (1 - torch.cos(y))) \
            - (2 * x * y * torch.sin(y))
    f = (trigTerm + (y ** 2) * expTerm) / ((y ** 2 + x ** 2)**2+epsilon)
    return f

def ConvBolusExp(t, mb, me):
    '''
    expets torch tensors, with constants:
    t: timesteps [T, B]
    mb: mb [1]
    me: me [1]
    '''
    tB = 2 * torch.pi / mb
    I1 = ((t >= 0) & (t < tB))
    I2 = (t >= tB)
    I1f = I1.to(dtype=t.dtype)
    I2f = I2.to(dtype=t.dtype)

    y = torch.zeros_like(t)

    y = torch.where(I1, (t*I1f*SpecialCosineExp(me*t, mb*t)), 0)
    y = torch.where(I2, tB * I2f * SpecialCosineExp(me * tB, mb * tB) * torch.exp(-1*I2f* me * (t - tB)), y)

    # y[I1] = (t*I1f * SpecialCosineExp(me*t, (mb * t)))[I1]
    # y[I2] = (tB * I2f * SpecialCosineExp(me * tB, mb * tB) * torch.exp(-1*I2f* me * (t - tB)))[I2]

    # assert torch.allclose(y, y2, atol=1e-5)
    return y # [T, B]

def SpecialCosineExp(x,y):
    epsilon = 1e-5
    x += epsilon
    y += epsilon

    expTerm = torch.where(x!=0, (1-torch.exp(-x)) / (x), 0)
    # convert nans to 0
    # expTerm = torch.nan_to_num(expTerm)
    trigTerm = x * (1-torch.cos(y)) - y * torch.sin(y)
    dn = x**2 + y**2 #+ epsilon
    f = torch.where(dn!=0., (trigTerm + y**2 * expTerm) / dn, 0)
    return f

def CosineBolus(t, m):
    '''
    expects torch tensors, with constants:
    t: timesteps [T, B]
    m: m [1]
    '''
    # create timesteps for each sample in the batch  
    z = m * t
    I = (z >= 0) & (z <= (2*torch.pi))
    y = torch.zeros_like(t)
    y = torch.where(I, 1 - torch.cos(z), 0)

    return y # [T, B]
