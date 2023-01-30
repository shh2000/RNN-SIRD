import numpy as np
import matplotlib.pyplot as plt

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

file = open('../u_result.txt', encoding='utf8')
df = {}

last_area = ''
last_date = ''
for line in file.readlines():
    if len(line) < 40:
        info = line.replace('\n', '').split(' ')
        area = info[0]
        date = info[1]
        if area not in df.keys():
            df[area] = {}
        if date not in df[area].keys():
            df[area][date] = []
        last_area = area
        last_date = date
    else:
        info = line[1:-2].split(' ')
        for item in info:
            df[last_area][last_date].append(float(item.replace(',', '')))

dates = {}
for area in df.keys():
    for date in df[area].keys():
        if date not in dates.keys():
            dates[date] = 0

result = {}
for date in dates.keys():
    result[date] = {}
    for area in df.keys():
        u_origin = np.array(df[area][date])
        tmp = np.diff(u_origin)[5:]
        maxv = np.abs(tmp)[0]
        result[date][area] = {'max': maxv}
        max_start = u_origin[5]
        min_end = u_origin[-1]
        edge = 0.1 * max_start + 0.9 * min_end
        for i in range(5, len(u_origin)):
            if u_origin[i] < edge:
                result[date][area]['speed'] = i - 5
                break

x_labels = []
x = []
date2index = {}
i = 1
for date in result.keys():
    x_labels.append(date)
    x.append(i)
    date2index[date] = i
    i += 1

colorpool = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
             'tab:pink', 'tab:gray', 'tab:olive', 'c', 'indigo']
area2color = {}
i = 0
for area in df.keys():
    area2color[area] = colorpool[i]
    i += 1
print(area2color)

plt.figure(figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Max Policy Speed')
plt.title('Max Policy Speed and Policy Influence')
whole_size = 3

plt.xticks(x, x_labels, rotation=30)

for area in df.keys():
    plt.scatter(x=1, y=0.07, s=50, color=area2color[area], label=areacn2en[area], alpha=0.8)
    for date in result.keys():
        plt.scatter(x=date2index[date], y=result[date][area]['max'], s=(result[date][area]['speed'] ** 3) * whole_size,
                    color=area2color[area], alpha=0.8, marker='_')

    plt.legend(bbox_to_anchor=(0.9, 1.0))
plt.text(2, 0.0, r'A longer line means it takes longer for the policy to take effect.',fontsize=12)
plt.scatter(x=1, y=0.07, s=80, c='white')
plt.savefig('../policy')
