import torch


class RNN_Model(torch.nn.Module):
    def __init__(self, u_length, recover_rate, death_rate):
        super(RNN_Model, self).__init__()
        self.u_length = u_length
        self.rnn = torch.nn.RNN(input_size=1, hidden_size=self.u_length, num_layers=1)
        self.recover_rate = recover_rate.unsqueeze(0)
        self.death_rate = death_rate.unsqueeze(0)

    def forward(self, input, sird):
        death_rate = self.death_rate.squeeze(-1)
        recover_rate = self.recover_rate.squeeze(-1)

        hidden = torch.zeros(1, 1, self.u_length)
        out_fake, _ = self.rnn(input, hidden)
        out = torch.relu(out_fake)
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
        hidden = torch.zeros(1, 1, self.u_length)
        out_fake, _ = self.rnn(input, hidden)
        out = torch.relu(out_fake)
        return out

    def probe_x(self, input, sird):
        death_rate = self.death_rate.squeeze(-1)
        recover_rate = self.recover_rate.squeeze(-1)

        hidden = torch.zeros(1, 1, self.u_length)
        out_fake, _ = self.rnn(input, hidden)
        out = torch.relu(out_fake)
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
    model = RNN_Model(5, recover_rate, death_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    optimizer.zero_grad()
    beta = torch.tensor([[[.3]]])
    sird = torch.tensor([[[9974., 10., 15., 1., 10000.]]])
    y_pred = model(beta, sird)
    y_label = torch.tensor([[[9934., 20., 44., 2., 10000.]]])
    loss = (y_label[0][0][-1] - y_pred[0][0][-1]) ** 2
    loss.backward()
    optimizer.step()
