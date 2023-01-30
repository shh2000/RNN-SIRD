from matplotlib import pyplot as plt

df = open('../dats/select.csv', encoding='gbk')
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

ms = {}
df.readline()
for i in range(21):
    line = df.readline()
    info = line.split('	')
    model = info[0] + '-' + info[1]
    el = info[2]
    if model not in ms.keys():
        ms[model] = {}
    ms[model][el] = []
    for item in info[3:-2]:
        ms[model][el].append(float(item))

ps = {}
df.readline()
df.readline()
for i in range(12):
    line = df.readline()
    info = line.split('	')
    tl = int(info[0])
    size = int(info[1])
    if tl not in ps.keys():
        ps[tl] = {}
    ps[tl][size] = []
    for item in info[2:-2]:
        if item == 'NaN':
            continue
        ps[tl][size].append(float(item))

ms_plt = {}
i = 1
for model in ms.keys():
    ms_plt[model] = {'index': i, 'data': []}
    for el in ms[model].keys():
        for item in ms[model][el]:
            ms_plt[model]['data'].append(item)
    i += 1

plt.figure(figsize=(12, 6))
plt.xlabel('Model')
plt.ylabel('Loss(Lower Means Better)')
plt.title('Model Selection Experiments')
plt.yscale('log')
x_labels = []
x = []
y = []
colorpool = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
             'tab:pink', 'tab:gray', 'tab:olive', 'c', 'indigo']
for model in ms_plt.keys():
    x.append(ms_plt[model]['index'])
    x_labels.append(model)
    y.append(ms_plt[model]['data'])

plt.xticks(x, x_labels, rotation=30)
ms_vio = plt.violinplot(y)
i = 0
for item in ms_vio['bodies']:
    item.set_facecolor(colorpool[i])
    item.set_edgecolor(colorpool[i])
    item.set_alpha(0.5)
    i += 1
plt.savefig('../MS')

ps_plt = {}
i = 1
for size in ps[3].keys():
    ps_plt[size] = {'index': i, 'data': []}
    for tl in ps.keys():
        for item in ps[tl][size]:
            ps_plt[size]['data'].append(item)
    i += 1
for key in ps_plt.keys():
    print(key)
    print(ps_plt[key])
plt.figure(figsize=(12, 6))
plt.xlabel('Hidden Size')
plt.ylabel('Loss(Lower Means Better)')
plt.title('Parameters Selection Experiments')
plt.yscale('log')
x_labels = []
x = []
y = []
colorpool = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
             'tab:pink', 'tab:gray', 'tab:olive', 'c', 'indigo']
for size in ps_plt.keys():
    x.append(ps_plt[size]['index'])
    x_labels.append(size)
    y.append(ps_plt[size]['data'])

plt.xticks(x, x_labels, rotation=30)
ps_vio = plt.violinplot(y)
i = 0
for item in ps_vio['bodies']:
    item.set_facecolor(colorpool[i])
    item.set_edgecolor(colorpool[i])
    item.set_alpha(0.5)
    i += 1
plt.savefig('../PS')