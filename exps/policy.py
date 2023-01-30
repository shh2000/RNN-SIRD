import datetime as dt
import numpy as np
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

for date in dates.keys():
    plt.figure()
    for area in df.keys():
        u_origin = np.array(df[area][date])
        plt.yscale('log')
        tmp = np.diff(u_origin)[5:]
        tmp = np.abs(tmp)
        plt.plot(tmp, label=areacn2en[area])
        plt.title(date + ' Controlled U Compare')
        plt.xlabel('days')
        plt.ylabel('u')
        plt.legend()
    plt.savefig('../pics/' + date + '_ud_merge_area')
    plt.close()
