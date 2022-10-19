import torch
from NN_models.RNN_step_model import RNN_step_Model
import datetime as dt
from matplotlib import pyplot as plt


def get_casedata(areaname, popu, start_date, days):
    cases = {}
    for kind in ['confirmed', 'deaths', 'recovered']:
        df = open('../casedata/' + kind + '_sum.csv')
        df.readline()
        for line in df.readlines():
            area = line.split(',')[0]
            if area == areaname:
                cases[kind] = line.split(',')[1:-1]
                break

    index = start_date - dt.date(2020, 1, 23)
    confirmed = cases['confirmed'][index.days: index.days + days]
    deaths = cases['deaths'][index.days: index.days + days]
    recovered = cases['recovered'][index.days: index.days + days]
    for i in range(len(confirmed)):
        value = float(confirmed[i]) - float(deaths[i]) - float(recovered[i])
        confirmed[i] = str(value)

    x = []
    for i in range(days):
        x.append([popu - float(confirmed[i]) - float(recovered[i]) - float(deaths[i]), float(confirmed[i]),
                  float(recovered[i]), float(deaths[i])])
    deltaS = []
    for i in range(days - 1):
        deltaS.append(x[i][0] - x[i + 1][0])
    beta_true = []
    for i in range(days - 1):
        beta_true.append(deltaS[i] * popu / (x[i][0] * x[i][1]))
    return confirmed, recovered, deaths, beta_true


## 基本配置
start_date = dt.date(2020, 3, 29)
days = 8
area = '意大利'
popu = 59550000
train = True
base_lr = 0.01

## 超参
base_recover_rate = 0.01
base_death_rate = 0.005
base_beta = 0.07
hidden_size = 1
lr_cycle = 500
lr_rate = 0.3
epoch = 4

## 获取ground_truth
u_length = days - 1
df = get_casedata(area, popu, start_date, days)
#### init
i = float(df[0][0])
r = float(df[1][0])
d = float(df[2][0])
sird_init_list = [popu - i - r - d, i, r, d, popu]
sird_init = torch.tensor([[sird_init_list]])
#### final
i = float(df[0][-1])
r = float(df[1][-1])
d = float(df[2][-1])
sird_final_list = [popu - i - r - d, i, r, d, popu]
sird_final = torch.tensor([[sird_final_list]])

recover_rate = torch.tensor([[base_recover_rate]])
death_rate = torch.tensor([[base_death_rate]])
beta_natural = torch.tensor([[[base_beta]]])

if train:
    model = RNN_step_Model(u_length, recover_rate, death_rate, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    for epoch in range(epoch * lr_cycle):
        if epoch % lr_cycle == 0 and epoch != 0:
            base_lr *= lr_rate
            optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

        optimizer.zero_grad()
        outputs = model(beta_natural, sird_init)

        i_loss = (sird_final[0][0][1] - outputs[0][0][1]) ** 2
        r_loss = (sird_final[0][0][2] - outputs[0][0][2]) ** 2
        d_loss = (sird_final[0][0][3] - outputs[0][0][3]) ** 2

        u = model.probe_u(beta_natural)
        u_loss_abs = u_length * 3.0 - u.sum()

        u_var_loss = torch.abs(torch.std(u) - 600.0)

        # loss = i_loss + r_loss + d_loss + 1e6 * u_loss_abs
        loss = i_loss + r_loss + d_loss + 1e5 * u_var_loss
        # loss = i_loss
        if epoch % 100 == 0:
            print(loss)
            print(torch.std(u))
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(),
               '../saved_model/RNNStep' + area + start_date.strftime('%Y%m%d') + '_' + str(days) + 'days')
else:
    model = RNN_step_Model(u_length, recover_rate, death_rate, hidden_size)
    model.load_state_dict(torch.load('../saved_model/RNNStep中国20200216_15days'))
    model.eval()
beta_pred = model.probe_u(beta_natural)
outputs = model(beta_natural, sird_init)

plt.plot(list(beta_pred[0][0]), 'r', label='Pred')
plt.plot(df[3], 'b', label='True')
plt.title('β value')
plt.xlabel('days')
plt.ylabel('β')
plt.legend()
plt.show()

plt.plot(list(outputs[0][0])[1:4], "r*:", label='Pred')
plt.plot(list(sird_final[0][0])[1:4], "b*:", label='True')
plt.xticks(range(3), ['Infectious', 'Recovered', 'Death'])
plt.title('Case Data Compare')
plt.ylabel('number')
plt.legend()
plt.show()

t_range_subdt = [start_date + dt.timedelta(days=x + 1) for x in range(days - 1)]

## true cases
newconf_true = []
newconf_pred = []
for i in range(days - 1):
    newconf_true.append(float(df[0][i + 1]) - float(df[0][i]))
print(newconf_true)
## pred cases
cases = model.probe_x(beta_natural, sird_init)
last = sird_init_list[1]
for item in cases:
    today = float(item[0][0][0])
    newconf_pred.append(today - last)
    last = today

plt.figure()
# plt.yscale('log')
plt.plot(t_range_subdt, newconf_pred, "r", label='Optimized Controlled Confirm Cases(compartment I)')
plt.plot(t_range_subdt, newconf_true, "b*:", label='Real Dailynew Confirm Cases(inout)')
plt.title('Case Data Compare')
plt.ylabel('number')
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()
