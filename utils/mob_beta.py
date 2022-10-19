import math
from matplotlib import pyplot as plt
import numpy as np
import torch


def beta2mob_single(beta, betamax, betamin, mobmax, mobmin, coe):
    v = (beta - betamin) / (betamax - betamin)
    u = (math.exp(coe * v) - 1) / (math.exp(coe) - 1)
    y = u * (mobmax - mobmin) + mobmin
    return y


def beta2mob_array(beta, betamax, betamin, mobmax, mobmin, coe):
    result = []
    for item in beta:
        result.append(beta2mob_single(item, betamax, betamin, mobmax, mobmin, coe))
    return result


def mob2beta_single(mob, betamax, betamin, mobmax, mobmin, coe):
    u = (mob - mobmin) / (mobmax - mobmin)
    v = math.log(u * (math.exp(coe) - 1) + 1) / coe
    x = v * (betamax - betamin) + betamin
    return x


def mob2beta_array(mob, betamax, betamin, mobmax, mobmin, coe):
    result = []
    for item in mob:
        result.append(mob2beta_single(item, betamax, betamin, mobmax, mobmin, coe))
    return result


def beta2mob_torch(beta, betamax, betamin, mobmax, mobmin, coe):
    v = (beta - betamin) / (betamax - betamin)
    u = (torch.exp(coe * v) - 1) / (math.exp(coe) - 1)
    y = u * (mobmax - mobmin) + mobmin
    return y


def beta2mob_torch_mac(beta, betamax, betamin, mobmax, mobmin, coe):
    v = (beta - betamin) / (betamax - betamin)
    u = (coe * v + (coe * v * coe * v) / 2 + (coe * v * coe * v * coe * v) / 6) / (math.exp(coe) - 1)
    y = u * (mobmax - mobmin) + mobmin
    return y


def beta2mob_torch_linear(beta, coe):
    return beta*coe


if __name__ == '__main__':
    beta = np.linspace(1.0, 4.0, 100)
    beta_torch = torch.tensor([[beta]])
    mob_torch = beta2mob_torch(beta_torch, 4.0, 1.0, 4.0, 1.0, 5.0)
    plt.plot(beta_torch[0][0], mob_torch[0][0])
    logmob = []
    for item in mob_torch[0][0]:
        logmob.append(math.log(item))
    plt.plot(beta_torch[0][0], logmob)
    plt.show()
