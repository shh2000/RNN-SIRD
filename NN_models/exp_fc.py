import torch


class EnDeModel(torch.nn.Module):
    def __init__(self, recover_rate, death_rate, history_length, future_u_length, hidden_size):
        super(EnDeModel, self).__init__()
        self.recover_rate = recover_rate
        self.death_rate = death_rate

        self.history_length = history_length
        self.future_u_length = future_u_length

        self.fc = torch.nn.Linear(self.history_length, self.future_u_length)

    def forward(self, history_u, sird, target_step):
        r = self.fc(history_u)

        betat = []

        for step in range(self.future_u_length):
            out = torch.sigmoid(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(r[0][0][step], 0), 0), 0))
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

        x_foo = []
        for item in range(target_step - 1):
            s, i, r, d, n = torch.split(x, 1, dim=2)
            si = s * i * betat[-1] / n
            ir = i * self.recover_rate
            id = i * self.death_rate
            r = r + ir
            d = d + id
            s = s - si
            i = i + si - ir - id
            x = torch.cat([s, i, r, d, n], 2)
            x_foo.append(x)

        return x, x_foo

    def probe_u(self, history_u):
        r = self.fc(history_u)

        betat = []

        for step in range(self.future_u_length):
            out = torch.sigmoid(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(r[0][0][step], 0), 0), 0))
            betat.append(out)

        u = torch.cat(betat, dim=2)
        return u

    def probe_x(self, history_u, sird):
        r = self.fc(history_u)

        betat = []

        for step in range(self.future_u_length):
            out = torch.sigmoid(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(r[0][0][step], 0), 0), 0))
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
