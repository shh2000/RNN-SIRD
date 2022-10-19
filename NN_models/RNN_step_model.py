import torch
from torchsummary import summary


class RNN_step_Model(torch.nn.Module):
    def __init__(self, u_length, recover_rate, death_rate, hidden_size):
        super(RNN_step_Model, self).__init__()
        self.u_length = u_length
        self.hidden_size = hidden_size
        self.encoder = torch.nn.Linear(1, self.hidden_size)
        self.rnn = torch.nn.GRU(input_size=hidden_size, hidden_size=self.hidden_size, num_layers=1)
        self.linear = torch.nn.Linear(self.hidden_size, 1)
        self.recover_rate = recover_rate
        self.death_rate = death_rate

    def forward(self, input, sird):

        h = torch.zeros(1, 1, self.hidden_size)
        betat = []
        y_1 = input

        for step in range(self.u_length):
            y = self.encoder(y_1)
            y, h = self.rnn(y, h)
            y_1 = torch.relu(self.linear(y)) + 0.02
            betat.append(y_1)

        x = sird
        for days in range(self.u_length):
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

    def probe_u(self, input):
        h = torch.zeros(1, 1, self.hidden_size)
        betat = []
        y_1 = input

        for step in range(self.u_length):
            y = self.encoder(y_1)
            y, h = self.rnn(y, h)
            y_1 = torch.relu(self.linear(y)) + 0.02
            betat.append(y_1)
        out = torch.cat(betat, dim=2)
        return out

    def probe_x(self, input, sird):
        h = torch.zeros(1, 1, self.hidden_size)
        betat = []
        y_1 = input

        for step in range(self.u_length):
            y = self.encoder(y_1)
            y, h = self.rnn(y, h)
            y_1 = torch.relu(self.linear(y)) + 0.02
            betat.append(y_1)
        x = sird
        result = []
        for days in range(self.u_length):
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
    model = RNN_step_Model(8, recover_rate, death_rate, 64)
    print(summary(model, input_size=[(1, 1), (1, 5)]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    optimizer.zero_grad()
    beta = torch.tensor([[[.3]]])
    sird = torch.tensor([[[9974., 10., 15., 1., 10000.]]])
    y_pred = model(beta, sird)
    y_label = torch.tensor([[[9934., 20., 44., 2., 10000.]]])
    loss = (y_label[0][0][-1] - y_pred[0][0][-1]) ** 2
    loss.backward()
    optimizer.step()
