"""
This file contains the numpy implementation of the Extended Tofts model and helper functions to convert between concentration and parameters, necessary for non-linear least squares fitting.
"""
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import torch

def nrmse(y_true, y_pred):
    '''
    Normalized root mean square error
    :param y_true: true values
    :param y_pred: predicted values
    :return: nrsme
    '''
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    return np.sqrt(np.sum((y_true - y_pred)**2, axis=-1) / (np.mean(y_true, axis=-1)))

def fit_tofts_model(Ct, t, aif, idxs=None, X0=(1.0, 0.5, 0.3, 0.03),
                    bounds=((1e-8, 1e-2, 1e-8, 1e-8), (5.0, 1.0, 1.0, 1.2)), jobs=64, model='Cosine4'):
    '''
    Solves the Extended Tofts model for each voxel and returns model parameters using the computationally efficient AIF as described by Orton et al. 2008 in https://doi.org/10.1088/0031-9155/53/5/005.
    :param Ct: Concentration curve as a N x T matrix, with N the number of curves to fit and T the number of time points
    :param t: Time samples at which Ct is measured
    :param aif: The aif model parameters in library, including ab, ae, mb, me and t0. can be obtained by fitting Cosine4AIF
    :param idxs: optional parameter determining which idxs to fit
    :param X0: initial guess of parameters ke, dt, ve, vp
    :param bounds: fit boundaries for  ke, dt, ve, vp
    :return output: matrix with EXtended Tofts parameters per voxel: ke,dt,ve,vp
    '''
    N, ndyn = Ct.shape
    if idxs is None:
        idxs = range(N)
    if model == 'Cosine4':
        fit_func = lambda tt, ke, dt, ve, vp: Cosine4AIF_ExtKety(tt, aif, ke, dt, ve, vp)
    popt_default = [1e-8, 1e-2, 1e-6, 1e-6]
    if len(idxs) < 2:
        try:
            output, pcov = curve_fit(fit_func, t, Ct[0], p0=X0, bounds=bounds)
        except:
            output = popt_default
            pcov = np.zeros((4, 4)) + 1e-8
        return output, pcov
    else:
        def parfun(idx, timing=t):
            if isnan(Ct[idx, :]).any() > 0:
                popt = popt_default
            else:
                try:
                    popt, pcov = curve_fit(fit_func, timing, Ct[idx, :], p0=X0, bounds=bounds)
                except RuntimeError:
                    popt = popt_default
                    pcov = np.zeros((4, 4))
            return popt, pcov
        out = Parallel(n_jobs=jobs, verbose=0)(delayed(parfun)(i) for i in idxs)
        means = [o[0] for o in out]
        covs = [o[1] for o in out]

        output = np.array(means)
        pcov = np.array(covs)
        return output, pcov
    
def Cosine4AIF_ExtKety(t, aif, ke, dt, ve, vp):
    # offset time array
    t = t - aif['t0'] - dt

    cpBolus = aif['ab'] * CosineBolus(t, aif['mb'])
    cpWashout = aif['ab'] * aif['ae'] * ConvBolusExp(t, aif['mb'], aif['me'])
    ceBolus = ke * aif['ab'] * ConvBolusExp(t, aif['mb'], ke)
    ceWashout = ke * aif['ab'] * aif['ae'] * ConvBolusExpExp(t, aif['mb'], aif['me'], ke)

    cp = cpBolus + cpWashout
    ce = ceBolus + ceWashout

    ct = np.zeros(np.shape(t))
    ct[t > 0] = vp * cp[t > 0] + ve * ce[t > 0]

    return ct

def CosineBolus(t, m):
    z = array(m * t)
    I = (z >= 0) & (z < (2 * pi))
    y = np.zeros(np.shape(t))
    y[I] = 1 - cos(z[I])

    return y


def ConvBolusExp(t, m, k):
    tB = 2 * pi / m
    tB = tB
    t = array(t)
    I1 = (t > 0) & (t < tB)
    I2 = t >= tB

    y = np.zeros(np.shape(t))

    y[I1] = multiply(t[I1], SpecialCosineExp(k * t[I1], m * t[I1]))
    y[I2] = tB * SpecialCosineExp(k * tB, m * tB) * exp(-k * (t[I2] - tB))

    return y


def ConvBolusExpExp(t, m, k1, k2):
    tol = 1e-4

    tT = tol / abs(k2 - k1)
    tT = array(tT)
    Ig = (t > 0) & (t < tT)
    Ie = t >= tT
    y = np.zeros(np.shape(t))

    y[Ig] = ConvBolusGamma(t[Ig], m, 0.5 * (k1 + k2))
    y1 = ConvBolusExp(t[Ie], m, k1)
    y2 = ConvBolusExp(t[Ie], m, k2)
    y[Ie] = (y1 - y2) / (k2 - k1)

    return y


def ConvBolusGamma(t, m, k):
    tB = 2 * pi / m
    tB = array(tB)
    y = np.zeros(np.shape(t))
    I1 = (t > 0) & (t < tB)
    I2 = t >= tB

    ce = SpecialCosineExp(k * tB, m * tB)
    cg = SpecialCosineGamma(k * tB, m * tB)

    y[I1] = square(t[I1]) * SpecialCosineGamma(k * t[I1], m * t[I1])
    y[I2] = tB * multiply(((t[I2] - tB) * ce + tB * cg), exp(-k * (t[I2] - tB)))

    return y


def SpecialCosineGamma(x, y):
    x = array(x)
    y = array(y)
    x2 = square(x)
    y2 = square(y)
    expTerm = multiply(3 + divide(square(y), square(x)), (1 - exp(-x))) \
              - multiply(divide(square(y) + square(x), x), exp(-x))
    trigTerm = multiply((square(x) - square(y)), (1 - cos(y))) - multiply(multiply(2 * x, y), sin(y))
    f = divide((trigTerm + multiply(square(y), expTerm)), square(square(y) + square(x)))

    return f


def SpecialCosineExp(x, y):
    x = array(x)
    y = array(y)
    expTerm = divide((1 - exp(-x)), x)
    # convert nans to 0
    expTerm = np.nan_to_num(expTerm)
    trigTerm = multiply(x, (1 - cos(y))) - multiply(y, sin(y))
    f = divide((trigTerm + multiply(square(y), expTerm)), (square(x) + square(y)))
    return f


# some helper functions to simulate data
def con_to_R1eff(C, R1map, relaxivity):
    assert (relaxivity > 0.0)
    return R1map + relaxivity * C


def r1eff_to_dce(R, TR, flip):
    S = ((1 - exp(-TR * R)) * sin(flip)) / (1 - cos(flip) * exp(-TR * R))
    return S


def R1_two_fas(images, flip_angles, TR):
    ''' Create T1 map from multiflip images '''
    inshape = images.shape
    nangles = inshape[-1]
    n = np.prod(inshape[:-1])
    images = np.reshape(images, (n, nangles))
    # flip_angles = pi*arange(20,0,-2)/180.0  # deg
    assert (nangles == 2)
    assert (len(flip_angles) == 2)
    signal_scale = abs(images).max()
    images = images / signal_scale
    R1map = zeros(n)
    c1 = cos(flip_angles[0])
    c2 = cos(flip_angles[1])
    s1 = sin(flip_angles[0])
    s2 = sin(flip_angles[1])
    rho = images[:, 1] / images[:, 0]
    for j in range(n):
        if images[j, :].mean() > 0.05:
            try:
                R1map[j] = np.log((rho[j] * s1 * c2 - c1 * s2) / (rho[j] * s1 - s2)) / TR
                #:https://iopscience.iop.org/article/10.1088/0031-9155/54/1/N01/meta
                #               R1map[j] = TR * 1/(np.log((images[j,0] * c1 * s2 - images[j,1] * s1 * c2) /
                #                          (images[j,0] * s2 - images[j,1] * s1)))
            except RuntimeError:
                R1map[j] = 0
                print(j)
    return (R1map)


def dce_to_r1eff(S, S0, R1, TR, flip):
    # taken from https://github.com/welcheb/pydcemri/blob/master from David S. Smith
    # Copyright (C) 2014   David S. Smith
    #
    # This program is free software; you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation; either version 2 of the License, or
    # (at your option) any later version.
    #
    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.
    #
    # You should have received a copy of the GNU General Public License along
    # with this program; if not, write to the Free Software Foundation, Inc.,
    # 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
    #
    print('converting DCE signal to effective R1')
    assert (flip > 0.0)
    assert (TR > 0.0 and TR < 1.0)
    #S = S.T
    #S0 = np.repeat(np.expand_dims(S0, axis=1), len(S), axis=1).T
    A = S / S0  # normalize by pre-contrast signal
    E0 = exp(-R1 * TR)
    E = (1.0 - A + A * E0 - E0 * cos(flip)) / \
        (1.0 - A * cos(flip) + A * E0 * cos(flip) - E0 * cos(flip))

    R = (-1.0 / TR) * log(E)

    return R.T 


def r1eff_to_conc(R1eff, R1map, relaxivity):
    return (R1eff - R1map) / relaxivity


def signal2concentration(data, tr, t10, alpha, relaxivity, contrast_entry, mode='batch'):
    '''
    expects a single batch or voxel of data
    parameters:
    data: 2D tensor of shape (B,T)
    tr: repetition time
    t10: T1 of tissue
    alpha: flip angle (degrees)
    relaxivity: relaxivity of contrast agent (1/(mMs))
    contrast_entry: timestep at which contrast agent enters tissue
    mode: 'volume' or 'voxel'
    '''

    # if mode == 'volume':
    batch = data
    B,T = data.shape
    # convert to radians
    alphadeg = alpha
    alpha = alpha * np.pi / 180

    #convert to per mmol per ms
    relaxivity = relaxivity * 1e-3

    # calculate m0
    m0 = torch.mean(batch[:, 0:contrast_entry], axis=-1) 
    e1 = np.exp(-tr / t10)
    m0e = m0 * (1 - e1 * np.cos(alpha))
    m0e = m0e / (np.sin(alpha) * (1 - e1))

    # calculate t1
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    ct = torch.empty_like(batch)
    # loop to prevent kernel from crashing on cpu
    for t in range(T):
        lg = (m0e * sa - batch[:,t] * ca) / (m0e * sa - batch[:,t])
        mini = np.exp(tr/t10)+1e-6
        lg = torch.where(lg > mini, lg, mini)     
        t1 =  tr / torch.log(lg)
        ct[:,t] = 1 / relaxivity * (1 / t1 - 1 / t10)
        if torch.isnan(ct).sum().item() > 0:
            print('nan encountered and set to 0')
            ct[torch.isnan(ct)] = 0

    torch.cuda.empty_cache()
    return ct