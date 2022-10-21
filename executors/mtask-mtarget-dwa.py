import torch
from NN_models.seq2seq_multipred import EnDeModel
import datetime as dt
from matplotlib import pyplot as plt

areacn2en = {
    "美国": "U.S.",
    '巴西': "Brzail",
    '俄罗斯': "Russia",
    '日本': "Japan",
    '德国': "Deutschland",
    '法国': "France",
    '英国': "U.K.",
    '意大利': "Italy",
    '韩国': "South Korea",
    '波兰': "Poland",
    '澳大利亚': "Australia",
}


def dwa(L_old, L_new, T=2):
    """
    L_old: list.
    """
    L_old = torch.tensor(L_old, dtype=torch.float32)
    L_new = torch.tensor(L_new, dtype=torch.float32)
    N = len(L_new)  # task number
    r = L_old / L_new

    w = N * torch.softmax(r / T, dim=0)
    return w.numpy()


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


settings = open('../exp_settings.csv', encoding='utf8')
settings.readline()
for line in settings.readlines():
    torch.manual_seed(17373507)
    ## 基本配置

    future_days = 14  # 算上了今天这一天。例如，start选择为4.10，history=5，future=8，则history的日期为5-9，future的日期为10-17，betatrue的对应下标为6-10
    history_days = 5
    base_lr = 0.01

    ## 超参

    hidden_size = 16
    lr_cycle = 200
    lr_rate = 0.7
    epoch_size = 20
    target_step = 4

    info = line.replace('\n', '').split(',')
    y = int(info[1])
    m = int(info[2])
    d = int(info[3])

    area = info[0]
    start_date = dt.date(y, m, d)
    print(area, start_date)

    popu = int(info[4])
    base_recover_rate = float(info[6])
    base_death_rate = float(info[5])

    ## 获取ground_truth
    u_length = future_days - 1
    df = get_casedata(area, popu, start_date, history_days, future_days + target_step - 1)
    print(df[3])

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
        i = float(df[0][item - target_step + 1])
        r = float(df[1][item - target_step + 1])
        d = float(df[2][item - target_step + 1])
        sird_final_list = [popu - i - r - d, i, r, d, popu]
        sird_final_foo_item = torch.tensor([[sird_final_list]])
        sird_final_foo.append(sird_final_foo_item)

    df = get_casedata(area, popu, start_date, history_days, future_days)

    recover_rate = torch.tensor([[base_recover_rate]])
    death_rate = torch.tensor([[base_death_rate]])
    beta_history = torch.tensor([[df[3][:history_days]]])

    model = EnDeModel(recover_rate, death_rate, history_days, u_length, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    for epoch in range(epoch_size * lr_cycle):
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
        u_loss_abs = -u.sum()

        ustart = float(df[3][history_days - 1])
        uend = float(df[3][history_days + future_days - 2])
        weights = torch.arange(ustart, uend, (uend - ustart) * 1.00001 / u_length)
        u_policy_loss = ((u - weights) ** 2).sum()

        loss_t = [float(i_loss), float(u_loss_abs), float(u_policy_loss)]

        if (epoch == 0) or (epoch == 1):
            loss_tm1 = loss_t
            loss_weights = dwa(loss_tm1, loss_tm1)
        else:
            loss_tm2 = loss_tm1
            loss_tm1 = loss_t
            loss_weights = dwa(loss_tm1, loss_tm2)

        if epoch % 300 == 0:
            print(i_loss, u_loss_abs, u_policy_loss, loss_weights, ustart, uend)

        loss = i_loss * loss_weights[0] + u_loss_abs * loss_weights[1] + u_policy_loss * loss_weights[2]
        loss.backward()
        optimizer.step()
    """
    print('@@@@@')
    print(i_loss / target_step, u_loss_abs, u_policy_loss)
    print('*****')
    """
    str_file = open('../dats/' + area + '_' + start_date.strftime('%Y%m%d') + '_loss.log', 'w', encoding='utf8')
    str_file.write('Prediction loss: ')
    str_file.write(str(round(float(i_loss / target_step), 5)))
    str_file.write('\n')
    str_file.write('Finance abs loss: ')
    str_file.write(str(round(float(u_loss_abs), 5)))
    str_file.write('\n')
    str_file.write('Finance var loss: ')
    str_file.write(str(round(float(u_policy_loss), 5)))
    str_file.write('\n')

    beta_pred = torch.cat([beta_history, model.probe_u(beta_history)], dim=2)
    outputs, foo = model(beta_history, sird_init, target_step)

    u = torch.tensor(df[3][history_days:])
    u_loss_abs = - u.sum()

    ustart = float(df[3][history_days - 1])
    uend = float(df[3][history_days + future_days - 2])
    weights = torch.arange(ustart, uend, (uend - ustart) * 1.00001 / u_length)
    u_policy_loss = ((u - weights) ** 2).sum()
    # print(u_loss_abs, u_policy_loss)
    str_file.write('True abs loss: ')
    str_file.write(str(round(float(u_loss_abs), 5)))
    str_file.write('\n')
    str_file.write('True var loss: ')
    str_file.write(str(round(float(u_policy_loss), 5)))
    str_file.write('\n')
    str_file.close()

    plt.figure()
    plt.plot(list(beta_pred[0][0].detach().numpy()), 'r', label='Pred', linewidth=4)
    plt.plot(df[3], 'b', label='True', linewidth=2)
    plt.title(areacn2en[area] + ' Controlled U Compare')
    plt.xlabel('days')
    plt.ylabel('u')
    plt.legend()
    plt.savefig('../pics/' + area + '_' + start_date.strftime('%Y%m%d') + '_u')
    plt.close()

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
    plt.plot(t_range_subdt, newconf_pred, "r", label='Optimized Controlled I compartment')
    plt.plot(t_range_subdt, newconf_true, "b*:", label='Real Dailynew I compartment')
    plt.title(areacn2en[area] + ' Dailynew I compartment Compare')
    plt.ylabel('number')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.savefig('../pics/' + area + '_' + start_date.strftime('%Y%m%d') + '_x')
    plt.close()

    t_range_subdt = [start_date + dt.timedelta(days=x + 1) for x in range(future_days + target_step - 1)]
    wholeconf_pred = []
    wholeconf_true = []
    for i in range(future_days):
        wholeconf_true.append(float(df[0][i]))
    cases = model.probe_x(beta_history, sird_init)
    wholeconf_pred = [sird_init_list[1]]
    for item in cases:
        wholeconf_pred.append(float(item[0][0][0]))
    pred_target = []
    ground_truth = []
    for item in range(target_step - 1):
        sird_final_foo_item = sird_final_foo[item]
        foo_item = foo[item]
        ground_truth.append(float(sird_final_foo_item[0][0][1]))
        pred_target.append(float(foo_item[0][0][1]))

    plt.figure()
    # plt.yscale('log')
    plt.plot(t_range_subdt, wholeconf_pred + pred_target, "r", label='Optimized Controlled I compartment')
    plt.plot(t_range_subdt, wholeconf_true + ground_truth, "b*:", label='Real Dailynew I compartment')
    plt.axvline(t_range_subdt[future_days - 1], color='g', linestyle='--',
                label='Predict Edge(before: predict, after: target)')
    plt.title(areacn2en[area] + ' Accumulated I compartment Compare')
    plt.ylabel('number')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.savefig('../pics/' + area + '_' + start_date.strftime('%Y%m%d') + '_whole')
    plt.close()
