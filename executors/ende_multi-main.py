import torch
from NN_models.seq2seq_multipred import EnDeModel
import datetime as dt
from matplotlib import pyplot as plt
from math import sqrt


def get_casedata(areaname, popu, start_date, history_days, future_days):
    cases = {}
    for kind in ['confirmed', 'deaths', 'recovered']:
        df = open('../casedata/' + kind + '_sum.csv')
        df.readline()
        for line in df.readlines():
            area = line.split(',')[0]
            if area == areaname:
                cases[kind] = line.split(',')[1:-1]
                break

    whole_days = history_days + future_days

    index = start_date - dt.date(2020, 1, 23) - dt.timedelta(history_days)
    confirmed = cases['confirmed'][index.days: index.days + whole_days]
    deaths = cases['deaths'][index.days: index.days + whole_days]
    recovered = cases['recovered'][index.days: index.days + whole_days]
    for i in range(len(confirmed)):
        value = float(confirmed[i]) - float(deaths[i]) - float(recovered[i])
        confirmed[i] = str(value)

    x = []
    for i in range(whole_days):
        x.append([popu - float(confirmed[i]) - float(recovered[i]) - float(deaths[i]), float(confirmed[i]),
                  float(recovered[i]), float(deaths[i])])
    deltaS = []
    for i in range(whole_days - 1):
        deltaS.append(x[i][0] - x[i + 1][0])
    beta_true = []
    for i in range(whole_days - 1):
        beta_true.append(deltaS[i] * popu / (x[i][0] * x[i][1]))
    return confirmed[history_days:], recovered[history_days:], deaths[history_days:], beta_true


torch.manual_seed(17373507)
## 基本配置
start_date = dt.date(2020, 4, 1)
future_days = 14  # 算上了今天这一天。例如，start选择为4.10，history=5，future=8，则history的日期为5-9，future的日期为10-17，betatrue的对应下标为6-10
history_days = 7
area = '美国'
popu = 57000000
base_lr = 0.01

## 超参
base_recover_rate = 0.018
base_death_rate = 0.008
hidden_size = 128
lr_cycle = 200
lr_rate = 0.7
epoch = 20
target_step = 3

## 获取ground_truth
u_length = future_days - 1
df = get_casedata(area, popu, start_date, history_days, future_days + target_step - 1)

#### init
i = float(df[0][0])
r = float(df[1][0])
d = float(df[2][0])
sird_init_list = [popu - i - r - d, i, r, d, popu]
sird_init = torch.tensor([[sird_init_list]])
#### final
i = float(df[0][-target_step])
r = float(df[1][-target_step])
d = float(df[2][-target_step])
sird_final_list = [popu - i - r - d, i, r, d, popu]
sird_final = torch.tensor([[sird_final_list]])

sird_final_foo = []
for item in range(target_step - 1):
    #### multi1
    i = float(df[0][-2])
    r = float(df[1][-2])
    d = float(df[2][-2])
    sird_final_list = [popu - i - r - d, i, r, d, popu]
    sird_final_foo_item = torch.tensor([[sird_final_list]])
    sird_final_foo.append(sird_final_foo_item)

df = get_casedata(area, popu, start_date, history_days, future_days)

recover_rate = torch.tensor([[base_recover_rate]])
death_rate = torch.tensor([[base_death_rate]])
beta_history = torch.tensor([[df[3][:history_days]]])

model = EnDeModel(recover_rate, death_rate, history_days, u_length, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

for epoch in range(epoch * lr_cycle):
    if epoch % lr_cycle == 0 and epoch != 0:
        base_lr *= lr_rate
        for params in optimizer.param_groups:
            params['lr'] *= lr_rate

    optimizer.zero_grad()
    outputs, foo = model(beta_history, sird_init, target_step)

    i_loss_origin = (sird_final[0][0][1] - outputs[0][0][1]) ** 2

    i_loss = i_loss_origin
    for item in range(target_step - 1):
        sird_final_foo_item = sird_final_foo[item]
        foo_item = foo[item]
        i_loss += (sird_final_foo_item[0][0][1] - foo_item[0][0][1]) ** 2

    i_loss = torch.sqrt(i_loss)

    u = model.probe_u(beta_history)
    u_loss_abs = torch.abs(u_length * 0.2 - u.sum())

    weights = torch.arange(0.2, 0.0, -0.2 / u_length)
    u_policy_loss = ((u - weights) ** 2).sum()

    loss = i_loss + 1e1 * u_loss_abs + 1e3 * u_policy_loss
    if epoch % 100 == 0:
        print(loss, u_loss_abs, u_policy_loss)
        _ = 0
    loss.backward()
    optimizer.step()
print('@@@@@')
print(i_loss / target_step, u_loss_abs, u_policy_loss)
print('*****')

beta_pred = torch.cat([beta_history, model.probe_u(beta_history)], dim=2)
outputs, foo = model(beta_history, sird_init, target_step)

u = torch.tensor(df[3][history_days:])
u_loss_abs = torch.abs(u_length * 0.2 - u.sum())

weights = torch.arange(0.2, 0.0, -0.2 / u_length)
u_policy_loss = ((u - weights) ** 2).sum()
print(u_loss_abs, u_policy_loss)

plt.plot(list(beta_pred[0][0].detach().numpy()), 'r', label='Pred', linewidth=4)
plt.plot(df[3], 'b', label='True', linewidth=2)
plt.title('β value')
plt.xlabel('days')
plt.ylabel('β')
plt.legend()
plt.show()

plt.plot(list(outputs[0][0].detach().numpy())[1:4], "r*:", label='Pred')
plt.plot(list(sird_final[0][0].detach().numpy())[1:4], "b*:", label='True')
p = list(outputs[0][0].detach().numpy())[1:4]
t = list(sird_final[0][0].detach().numpy())[1:4]
for i in range(3):
    print(str(sird_init_list[i + 1]) + '/' + str(float(t[i])) + '/' + str(float(p[i])))
    print(round(abs(float(t[i]) - float(p[i])) / float(t[i]) * 100, 2))
plt.xticks(range(3), ['Infectious', 'Recovered', 'Death'])
plt.title('Case Data Compare')
plt.ylabel('number')
plt.legend()
plt.show()

t_range_subdt = [start_date + dt.timedelta(days=x + 1) for x in range(future_days - 1)]

## true cases
newconf_true = []
newconf_pred = []
for i in range(future_days - 1):
    newconf_true.append(float(df[0][i + 1]) - float(df[0][i]))

## pred cases
cases = model.probe_x(beta_history, sird_init)
last = sird_init_list[1]
for item in cases:
    today = float(item[0][0][0])
    newconf_pred.append(today - last)
    last = today

plt.figure()
# plt.yscale('log')
plt.plot(t_range_subdt, newconf_pred, "r", label='Optimized Controlled Confirm Cases')
plt.plot(t_range_subdt, newconf_true, "b*:", label='Real Dailynew Confirm Cases')
plt.title('Case Data Compare')
plt.ylabel('number')
plt.gcf().autofmt_xdate()
plt.legend()
plt.savefig('../pics/control_system(X)')
