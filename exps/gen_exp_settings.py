import datetime as dt
import numpy as np

np.set_printoptions(suppress=True)
area2popu = {
    "美国": 328000000,
    '巴西': 209000000,
    '俄罗斯': 146000000,
    '日本': 126000000,
    '德国': 82000000,
    '法国': 67000000,
    '英国': 66000000,
    '意大利': 60000000,
    '韩国': 51000000,
    '波兰': 38000000,
    '澳大利亚': 25000000,
}
dates = ['20200401', '20200515', '20200701', '20200815', '20201001', '20201115']
settings = open('../exp_settings.csv', 'w', encoding='utf8')
settings.write('area,year,month,day,popu,death,recover\n')

for area in area2popu.keys():
    for date in dates:
        str_info = area
        y, m, d = date[0:4], date[4:6], date[6:8]
        str_info += ',' + str(y)
        str_info += ',' + str(m)
        str_info += ',' + str(d)
        str_info += ',' + str(area2popu[area])

        start_date = dt.date(int(y), int(m), int(d))
        days = 30
        cases = {}
        for kind in ['confirmed', 'deaths', 'recovered']:
            df = open('../casedata/' + kind + '_sum.csv')
            df.readline()
            for line in df.readlines():
                areaii = line.split(',')[0]
                if areaii == area:
                    cases[kind] = line.split(',')[1:-1]
                    break

        index = start_date - dt.date(2020, 1, 23)
        confirmed = cases['confirmed'][index.days: index.days + days]
        deaths = cases['deaths'][index.days: index.days + days]
        recovered = cases['recovered'][index.days: index.days + days]
        for i in range(len(confirmed)):
            value = float(confirmed[i]) - float(deaths[i]) - float(recovered[i])
            confirmed[i] = str(value)

        rrate = []
        drate = []
        for i in range(days - 1):
            newr = float(recovered[i + 1]) - float(recovered[i])
            rrate.append(newr / float(confirmed[i]))
            if rrate[-1] < 0:
                rrate[-1] = 0
            newd = float(deaths[i + 1]) - float(deaths[i])
            drate.append(newd / float(confirmed[i]))
            if drate[-1] > 0.1:
                drate[-1] = drate[-2]
        rr = np.mean(np.array(rrate))
        dr = np.mean(np.array(drate))

        str_info += ',' + str('{:f}'.format(float(rr)))
        str_info += ',' + str('{:f}'.format(float(dr)))

        str_info += '\n'
        settings.write(str_info)
