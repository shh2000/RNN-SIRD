import torch


class EnDeModel(torch.nn.Module):
    def __init__(self, recover_rate, death_rate, history_length, future_u_length, hidden_size):
        super(EnDeModel, self).__init__()
        self.recover_rate = recover_rate
        self.death_rate = death_rate

        self.history_length = history_length
        self.future_u_length = future_u_length
        self.hidden_size = hidden_size

        self.linear_before_enc = torch.nn.Linear(1, self.hidden_size).cuda()
        self.enc_rnn = torch.nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1).cuda()
        self.dec_rnn = torch.nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1).cuda()
        self.linear_after_dec = torch.nn.Linear(self.hidden_size, 1).cuda()

    def forward(self, history_u, sird, target_step):
        h_enc = torch.zeros(1, 1, self.hidden_size).cuda()

        for step in range(self.history_length):
            u_tmp = history_u[0][0][step]
            u_tmp = torch.unsqueeze(u_tmp, 0)
            u_tmp = torch.unsqueeze(u_tmp, 0)
            u_tmp = torch.unsqueeze(u_tmp, 0)
            y = self.linear_before_enc(u_tmp)
            y, h_enc = self.enc_rnn(y, h_enc)
            h_enc = torch.sigmoid(h_enc)
        y = y
        h_dec = h_enc

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
        h_enc = torch.zeros(1, 1, self.hidden_size).cuda()

        for step in range(self.history_length):
            u_tmp = history_u[0][0][step]
            u_tmp = torch.unsqueeze(u_tmp, 0)
            u_tmp = torch.unsqueeze(u_tmp, 0)
            u_tmp = torch.unsqueeze(u_tmp, 0)
            y = self.linear_before_enc(u_tmp)
            y, h_enc = self.enc_rnn(y, h_enc)
            h_enc = torch.sigmoid(h_enc)
        y = y
        h_dec = h_enc

        betat = []

        for step in range(self.future_u_length):
            y, h_dec = self.dec_rnn(y, h_dec)
            out = torch.sigmoid(self.linear_after_dec(y))
            betat.append(out)

        u = torch.cat(betat, dim=2)
        return u

    def probe_x(self, history_u, sird):
        h_enc = torch.zeros(1, 1, self.hidden_size).cuda()

        for step in range(self.history_length):
            u_tmp = history_u[0][0][step]
            u_tmp = torch.unsqueeze(u_tmp, 0)
            u_tmp = torch.unsqueeze(u_tmp, 0)
            u_tmp = torch.unsqueeze(u_tmp, 0)
            y = self.linear_before_enc(u_tmp)
            y, h_enc = self.enc_rnn(y, h_enc)
            h_enc = torch.sigmoid(h_enc)
        y = y
        h_dec = h_enc

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
    recover_rate = 0.3
    death_rate = 0.01
    model = EnDeModel(recover_rate, death_rate, 5, 10, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    beta = torch.tensor([[[.25, .2, .3, .4, .46]]]).cuda()
    sird = torch.tensor([[[9974., 10., 15., 1., 10000.]]]).cuda()
    y_label = torch.tensor([[[9757., 98., 139., 5., 10000.]]]).cuda()
    for i in range(700):
        y_pred, foo = model(beta, sird, 1)
        loss = (y_label[0][0][1] - y_pred[0][0][1]) ** 2
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            print(loss)
            for params in optimizer.param_groups:
                params['lr'] *= 0.7

            print(y_pred)
