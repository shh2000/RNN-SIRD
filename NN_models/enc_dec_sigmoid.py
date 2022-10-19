import torch


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
        y = torch.sigmoid(self.enc(history_u))

        betat = []

        for step in range(self.future_u_length):
            y, h_dec = self.dec_rnn(y, h_dec)
            out = torch.sigmoid(self.linear_after_dec(y))
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
        y = torch.sigmoid(self.enc(history_u))

        betat = []

        for step in range(self.future_u_length):
            y, h_dec = self.dec_rnn(y, h_dec)
            out = torch.sigmoid(self.linear_after_dec(y))
            betat.append(out)

        u = torch.cat(betat, dim=2)
        return u

    def probe_x(self, history_u, sird):
        h_dec = torch.zeros(1, 1, self.hidden_size)
        y = torch.sigmoid(self.enc(history_u))

        betat = []

        for step in range(self.future_u_length):
            y, h_dec = self.dec_rnn(y, h_dec)
            out = torch.sigmoid(self.linear_after_dec(y))
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


if __name__ == '__main__':
    recover_rate = torch.tensor([0.01])
    death_rate = torch.tensor([0.03])
    model = EnDeModel(recover_rate, death_rate, 5, 8, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    optimizer.zero_grad()
    beta = torch.tensor([[[.3, .2, .3, .15, .4]]])
    sird = torch.tensor([[[9974., 10., 15., 1., 10000.]]])
    y_pred = model(beta, sird)
    y_label = torch.tensor([[[9934., 20., 44., 2., 10000.]]])
    loss = (y_label[0][0][-1] - y_pred[0][0][-1]) ** 2
    loss.backward()
    optimizer.step()
