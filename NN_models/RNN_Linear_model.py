import torch


class RNN_Linear_Model(torch.nn.Module):
    def __init__(self, u_length, recover_rate, death_rate, integrate_scale):
        super(RNN_Linear_Model, self).__init__()
        self.u_length = u_length
        self.integrate_scale = integrate_scale
        self.rnn = torch.nn.RNN(input_size=1, hidden_size=self.u_length * self.integrate_scale, num_layers=1)
        self.linear = torch.nn.Linear(self.u_length * self.integrate_scale, self.u_length)
        self.recover_rate = recover_rate.unsqueeze(0)
        self.death_rate = death_rate.unsqueeze(0)

    def forward(self, input, sird):
        death_rate = self.death_rate.squeeze(-1)
        recover_rate = self.recover_rate.squeeze(-1)

        hidden = torch.zeros(1, 1, self.u_length * self.integrate_scale)
        middle, _ = self.rnn(input, hidden)
        out = self.linear(middle)
        betat = torch.split(out, 1, dim=2)

        x = sird
        for days in range(self.u_length):
            s, i, r, d, n = torch.split(x, 1, dim=2)
            si = s * i * betat[days] / n
            ir = i * recover_rate
            id = i * death_rate
            r = r + ir
            d = d + id
            s = s - si
            i = i + si - ir - id
            x = torch.cat([s, i, r, d, n], 2)
        return x

    def probe_u(self, input):
        hidden = torch.zeros(1, 1, self.u_length * self.integrate_scale)
        middle, _ = self.rnn(input, hidden)
        out = self.linear(middle)
        return out

    def probe_x(self, input, sird):
        death_rate = self.death_rate.squeeze(-1)
        recover_rate = self.recover_rate.squeeze(-1)

        hidden = torch.zeros(1, 1, self.u_length * self.integrate_scale)
        middle, _ = self.rnn(input, hidden)
        out = self.linear(middle)
        betat = torch.split(out, 1, dim=2)

        x = sird
        result = []
        for days in range(self.u_length):
            s, i, r, d, n = torch.split(x, 1, dim=2)
            si = s * i * betat[days] / n
            ir = i * recover_rate
            id = i * death_rate
            r = r + ir
            d = d + id
            s = s - si
            i = i + si - ir - id
            x = torch.cat([s, i, r, d, n], 2)
            result.append([i, r, d])
        return result


if __name__ == '__main__':
    recover_rate = torch.tensor([
        [0.01]
    ])
    death_rate = torch.tensor([
        [0.03]
    ])
    model = RNN_Linear_Model(5, recover_rate, death_rate, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    beta = torch.tensor([[[.3]]])
    sird = torch.tensor([[[9974., 10., 15., 1., 10000.]]])
    y_pred = model(beta, sird)
    y_label = torch.tensor([[[9934., 20., 44., 2., 10000.]]])
    loss = (y_label[0][0][-1] - y_pred[0][0][-1]) ** 2
    loss.backward()
    optimizer.step()
