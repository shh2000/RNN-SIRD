from matplotlib import pyplot as plt

for scenario in range(4):
    df = open('../eval_results.csv', encoding='utf8')
    title = df.readline().split(',')
    plt.figure()
    plt.title('Regression-' + title[scenario])
    i = 0
    colorpool = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    data = {}
    for line in df.readlines():
        info = line.split(',')
        if info[scenario] not in data.keys():
            data[info[scenario]] = {'color': colorpool[i], 'data': []}
            i += 1
        data[info[scenario]]['data'].append(info[4:7])

    plt.ylabel('Optimal NPI Time Required')
    plt.xlabel('Optimal\'s Mortality')

    for sce in data.keys():
        i = 0
        x = []
        y = []
        for item in data[sce]['data']:
            x.append(float(item[2]))
            y.append(float(item[1]))
        plt.scatter(x, y, color=data[sce]['color'], alpha=0.7, label=sce)
    plt.legend()

    plt.savefig('../pics/eval_tr_' + title[scenario])
