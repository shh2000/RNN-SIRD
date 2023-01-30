import torch
from torch import tensor


class EnDeModel(torch.nn.Module):
    def __init__(self, recover_rate, death_rate, history_length, future_u_length, hidden_size):
        super(EnDeModel, self).__init__()
        self.recover_rate = recover_rate
        self.death_rate = death_rate

        self.history_length = history_length
        self.future_u_length = future_u_length
        self.hidden_size = hidden_size

        self.enc = torch.nn.Linear(self.history_length, self.future_u_length)

    def forward(self, history_u, sird):
        y = self.enc(history_u)
        betat = torch.split(y, 1, dim=2)

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
        y = self.enc(history_u)
        betat = torch.split(y, 1, dim=2)

        u = torch.cat(betat, dim=2)
        return u

    def probe_x(self, history_u, sird):
        y = self.enc(history_u)
        betat = torch.split(y, 1, dim=2)

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


if __name__ == '__main__':
    r = {'large_city-most_vaccined-in_epidemic@high_mobility-strict_target': {
        'u': [0.004974928218871355, 0.9731883406639099, 0.9731821417808533, 0.18890877068042755, 0.00591638358309865,
              0.00014900206588208675, 0.00014962110435590148, 0.00014960826956667006, 0.00014965777518227696,
              0.00014973456563893706, 0.00014982598077040166, 0.00014992473006714135, 0.0001500274083809927,
              0.00015013258962426335],
        'ird': [[280770.28125, 6728389.0, 3242.1611328125], [347388.53125, 6742427.5, 4219.24169921875],
                [427052.8125, 6759797.0, 5428.15380859375], [426865.03125, 6781149.5, 6914.2978515625],
                [404739.65625, 6802493.0, 8399.7880859375], [383110.96875, 6822730.0, 9808.2822265625],
                [362638.15625, 6841885.5, 11141.5087890625], [343259.375, 6860017.5, 12403.4892578125],
                [324916.1875, 6877180.5, 13598.0322265625], [307553.1875, 6893426.5, 14728.740234375],
                [291118.09375, 6908804.0, 15799.025390625], [275561.25, 6923360.0, 16812.1171875],
                [260835.734375, 6937138.0, 17771.0703125], [246897.15625, 6950180.0, 18678.779296875]],
        'loss': 5562.9384765625}}
