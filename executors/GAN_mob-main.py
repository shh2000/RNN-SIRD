import torch
from matplotlib import pyplot as plt
import random
import math


class CaseBetaGen():
    def __init__(self, principal):
        self.principal = principal

    def generate(self, beta_series: list, init_sird: dict, parameters: dict = None):
        if self.principal == 'SIRD':
            if 'deathRate' not in parameters.keys() or 'recoverRate' not in parameters.keys():
                return None
            s = init_sird['s']
            i = init_sird['i']
            r = init_sird['r']
            d = init_sird['d']
            results = {'s': [s], 'i': [i], 'r': [r], 'd': [d]}
            n = s + i + r + d
            for beta in beta_series:
                si = s * i * beta / n
                ir = i * parameters['recoverRate']
                id = i * parameters['deathRate']
                r = r + ir
                d = d + id
                s = s - si
                i = i + si - ir - id
                results['s'].append(s)
                results['i'].append(i)
                results['r'].append(r)
                results['d'].append(d)
            return results


class EnDeModel(torch.nn.Module):
    def __init__(self, recover_rate, death_rate, history_length, future_u_length, hidden_size):
        super(EnDeModel, self).__init__()
        self.recover_rate = recover_rate
        self.death_rate = death_rate

        self.history_length = history_length
        self.future_u_length = future_u_length
        self.hidden_size = hidden_size

        self.enc = torch.nn.Linear(self.history_length, self.hidden_size)
        self.dec_rnn = torch.nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1)
        self.linear_after_dec = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, history_u, sird):
        h_dec = torch.zeros(1, 1, self.hidden_size)
        y = torch.tanh(self.enc(history_u))

        betat = []

        for step in range(self.future_u_length):
            y, h_dec = self.dec_rnn(y, h_dec)
            y = torch.tanh(y)
            out = torch.abs(self.linear_after_dec(y))
            betat.append(out)

        x = sird
        for days in range(self.future_u_length):
            s, i, r, d, n = torch.split(x, 1, dim=2)
            si = s * i * betat[days] / n
            ir = i * self.recover_rate
            id = i * self.death_rate
            r = r + ir
            d = d + id
            s = s - si
            i = i + si - ir - id
            x = torch.cat([s, i, r, d, n], 2)
        return x

    def probe_u(self, history_u):
        h_dec = torch.zeros(1, 1, self.hidden_size)
        y = torch.tanh(self.enc(history_u))

        betat = []

        for step in range(self.future_u_length):
            y, h_dec = self.dec_rnn(y, h_dec)
            y = torch.tanh(y)
            out = torch.abs(self.linear_after_dec(y))
            betat.append(out)

        u = torch.cat(betat, dim=2)
        return u

    def probe_x(self, history_u, sird):
        h_dec = torch.zeros(1, 1, self.hidden_size)
        y = torch.tanh(self.enc(history_u))

        betat = []

        for step in range(self.future_u_length):
            y, h_dec = self.dec_rnn(y, h_dec)
            y = torch.tanh(y)
            out = torch.abs(self.linear_after_dec(y))
            betat.append(out)

        x = sird
        result = []
        for days in range(self.future_u_length):
            s, i, r, d, n = torch.split(x, 1, dim=2)
            si = s * i * betat[days] / n
            ir = i * self.recover_rate
            id = i * self.death_rate
            r = r + ir
            d = d + id
            s = s - si
            i = i + si - ir - id
            x = torch.cat([s, i, r, d, n], 2)
            result.append([i, r, d])
        return result


def beta2mob_torch_mac(beta, betamax, betamin, mobmax, mobmin, coe):
    v = (beta - betamin) / (betamax - betamin)
    u = (coe * v + (coe * v * coe * v) / 2 + (coe * v * coe * v * coe * v) / 6) / (math.exp(coe) - 1)
    y = u * (mobmax - mobmin) + mobmin
    return y


def gen_beta(beta0, length):
    result = []
    for i in range(length):
        result.append((random.random() * 0.6 + 0.7) * beta0 * (0.8 ** i))
    return result


def gen_init_data(beta0, length, s, i, r, d, recoverRate, deathRate):
    gen = CaseBetaGen('SIRD')
    beta = gen_beta(beta0, length)
    cases = gen.generate(beta, {'s': s, 'i': i, 'r': r, 'd': d}, {'recoverRate': recoverRate, 'deathRate': deathRate})
    return beta, cases


def train(model, base_lr, lr_cycle, lr_rate, epoch, enc_length, pred_length, beta, cases, mob_beta_coe, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    s = cases['s'][enc_length]
    i = cases['i'][enc_length]
    r = cases['r'][enc_length]
    d = cases['d'][enc_length]
    sird_init_list = [s, i, r, d, s + i + r + d]
    sird_init = torch.tensor([[sird_init_list]])

    s = cases['s'][-1]
    i = cases['i'][-1]
    r = cases['r'][-1]
    d = cases['d'][-1]
    sird_final_list = [s, i, r, d, s + i + r + d]
    sird_final = torch.tensor([[sird_final_list]])

    beta_history = torch.tensor([[beta[:enc_length]]])
    outputs = None

    for step in range(epoch * lr_cycle):
        if epoch % lr_cycle == 0 and epoch != 0:
            base_lr *= lr_rate
            optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

        optimizer.zero_grad()
        outputs = model(beta_history, sird_init)

        i_loss = (sird_final[0][0][1] - outputs[0][0][1]) ** 2
        r_loss = (sird_final[0][0][2] - outputs[0][0][2]) ** 2
        d_loss = (sird_final[0][0][3] - outputs[0][0][3]) ** 2

        u = model.probe_u(beta_history)
        mob = beta2mob_torch_mac(u, float(torch.max(u)), float(torch.min(u)), 100.0, -100.0, mob_beta_coe)
        # mob = beta2mob_torch_linear(u, mob_beta_coe)
        mob_loss_abs = torch.abs((pred_length - 1) * 100.0 - mob.sum())
        weights = torch.arange(-70.0, -100.0, -30.0 / (pred_length - 1))
        mob_policy_loss = ((mob - weights) ** 2).sum()
        loss = i_loss + r_loss + d_loss + mob_loss_abs + mob_policy_loss

        """
        u_loss_abs = torch.abs((pred_length - 1) * 0.10 - u.sum())
        weights = torch.arange(0.10, 0., -0.10 / (pred_length - 1))
        u_policy_loss = ((u - weights) ** 2).sum()
        loss = i_loss + r_loss + d_loss + u_loss_abs + u_policy_loss
        """
        ## 0.18=beta_today,
        ## 0.=beta_final,仿真/真实数据：末位beta或0，应用：0
        ## beta_future=从today，beta到清零日，0连一条直线

        if verbose and step % 100 == 0:
            print(loss, mob_loss_abs, mob_policy_loss)

        loss.backward()
        optimizer.step()

    plt.figure()
    plt.plot(list(outputs[0][0])[1:4], "r*:", label='Pred')
    plt.plot(list(sird_final[0][0])[1:4], "b*:", label='True')
    plt.xticks(range(3), ['Infectious', 'Recovered', 'Death'])
    plt.title('Case Data Compare')
    plt.ylabel('number')
    plt.legend()
    plt.show()
    beta_pred = torch.cat([beta_history, model.probe_u(beta_history)], dim=2)
    mob_pred = beta2mob_torch_mac(beta_pred, float(torch.max(beta_pred)), float(torch.min(beta_pred)), 100.0, -100.0,
                                  mob_beta_coe)
    # mob_pred = beta2mob_torch_linear(beta_pred, mob_beta_coe)

    p = list(outputs[0][0])[1:4]
    t = list(sird_final[0][0])[1:4]
    for i in range(3):
        print(str(sird_init_list[i + 1]) + '/' + str(float(t[i])) + '/' + str(float(p[i])))
        print(round(abs(float(t[i]) - float(p[i])) / float(t[i]) * 100, 2))

    newconf_true = []
    newconf_pred = []
    for i in range(pred_length - 1):
        newconf_true.append(float(cases['i'][enc_length + i + 1]) - float(cases['i'][enc_length + i]))

    cases = model.probe_x(beta_history, sird_init)
    last = sird_init_list[1]
    for item in cases:
        today = float(item[0][0][0])
        newconf_pred.append(today - last)
        last = today

    return list(beta_pred[0][0]), beta, newconf_pred, newconf_true, list(mob_pred[0][0])


torch.manual_seed(17373507)
random.seed(17373507)

base_lr = 0.01
lr_cycle = 300
lr_rate = 0.3
epoch = 10

base_recover_rate = 0.01
base_death_rate = 0.008
mb_coe = 3.0

enc_length = 7
pred_length = 15
data_length = enc_length + pred_length - 1
hidden_size = 16

model = EnDeModel(base_recover_rate, base_death_rate, enc_length, pred_length - 1, hidden_size)
"""
i0 = 210406.0
n = 100000000.0
beta, cases = gen_init_data(0.7, data_length, n - 1.6 * i0, i0, 0.5 * i0, 0.1 * i0, base_recover_rate, base_death_rate)
train(model, base_lr, lr_cycle, lr_rate, epoch, enc_length, pred_length, beta, cases,mob_beta_coe, True)
"""
i0 = 892051.0
n = 100000000.0
beta, cases = gen_init_data(0.3, data_length, n - 1.6 * i0, i0, 0.5 * i0, 0.1 * i0, base_recover_rate, base_death_rate)
bp, bt, cp, ct, mob = train(model, base_lr, lr_cycle, lr_rate, epoch, enc_length, pred_length, beta, cases, mb_coe,
                            True)

plt.figure()
plt.plot(bp, 'r', label='Pred', linewidth=4)
plt.plot(bt, 'b', label='True', linewidth=2)
plt.title('β value')
plt.xlabel('days')
plt.ylabel('β')
plt.legend()
plt.show()

plt.figure()
plt.plot(bp, 'r', label='Pred β', linewidth=2)
plt.plot(mob, 'b', label='Pred Mobility', linewidth=2)
plt.title('β and mobility')
plt.xlabel('days')
plt.ylabel('β/mobility(-100.0~100.0)')
plt.legend()
plt.show()

plt.figure()
plt.plot(cp, "r", label='Optimized Controlled Confirm Cases(compartment I)')
plt.plot(ct, "b*:", label='Real Dailynew Confirm Cases(inout)')
plt.title('Case Data Compare')
plt.ylabel('number')
plt.legend()
plt.show()
