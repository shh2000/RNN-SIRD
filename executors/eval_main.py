import torch
from NN_models.eval_rnnrnn import EnDeModel
import json
import gc


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


torch.manual_seed(17373507)
settings = json.load(open('../eval_settings.json', encoding='utf8'))
result = {}
for scenario in settings.keys():

    base_lr = 0.1
    lr_cycle = 300
    lr_rate = 0.7
    epoch_size = 10

    sird_init = settings[scenario]['init_sird']
    sird_init.append(sird_init[0] + sird_init[1] + sird_init[2] + sird_init[3])
    init = torch.tensor([[sird_init]])

    target = []
    for item in settings[scenario]['target']:
        sird_target = item.copy()
        sird_target.append(sird_target[0] + sird_target[1] + sird_target[2] + sird_target[3])
        target.append(torch.tensor([[sird_target]]))

    u_gen = settings[scenario]['u_his']
    beta_history = torch.tensor([[u_gen[0:4]]])

    model = EnDeModel(0.05, 0.00348)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    for epoch in range(epoch_size * lr_cycle):
        if epoch % lr_cycle == 0 and epoch != 0:
            base_lr *= lr_rate
            for params in optimizer.param_groups:
                params['lr'] *= lr_rate

        optimizer.zero_grad()
        outputs, foo = model(beta_history, init)

        i_loss = (foo[0][0][0][1] - target[0][0][0][1]) ** 2 + (foo[1][0][0][1] - target[1][0][0][1]) ** 2 + (
                foo[2][0][0][1] - target[2][0][0][1]) ** 2 + (foo[3][0][0][1] - target[3][0][0][1]) ** 2

        i_loss = torch.sqrt(i_loss)

        u = model.probe_u(beta_history)
        u_loss_abs = -u.sum()

        ustart = float(u_gen[4])
        uend = float(u_gen[17])
        weights = torch.arange(ustart, uend, (uend - ustart) * 1.00001 / 14)
        u_policy_loss = ((u - weights) ** 2).sum()

        loss_t = [float(i_loss), float(u_loss_abs), float(u_policy_loss)]

        if (epoch == 0) or (epoch == 1):
            loss_tm1 = loss_t
            loss_weights = dwa(loss_tm1, loss_tm1)
        else:
            loss_tm2 = loss_tm1
            loss_tm1 = loss_t
            loss_weights = dwa(loss_tm1, loss_tm2)

        loss = i_loss * loss_weights[0] + u_loss_abs * loss_weights[1] + u_policy_loss * loss_weights[2]
        loss.backward()
        optimizer.step()

    u_result = model.probe_u(beta_history)
    x_result = model.probe_x(beta_history, init)
    u = []
    x = []

    for item in u_result[0][0]:
        u.append(float(item))

    for item in x_result:
        i = float(item[0][0][0][0])
        r = float(item[1][0][0][0])
        d = float(item[2][0][0][0])
        x.append([i, r, d])
    result[scenario] = {'u': u, 'ird': x, 'loss': float(loss / 4.)}
    print(scenario)
    del model
    gc.collect()
    json.dump(result, open('../eval_results.json', 'w', encoding='utf8'))
