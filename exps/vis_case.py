import datetime as dt
from matplotlib import pyplot as plt

if __name__ == '__main__':
    areaname = '美国'
    popu = 57000000
    start_date = dt.date(2020, 5, 1)
    days = 30

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

    newconf = []
    for i in range(days - 1):
        newconf.append(float(confirmed[i + 1]) - float(confirmed[i]))
        if newconf[-1] > 5e5:
            newconf[-1] = newconf[-2]
    rrate = []
    drate = []
    for i in range(days - 1):
        newr = float(recovered[i + 1]) - float(recovered[i])
        rrate.append(newr / float(confirmed[i]))
        if rrate[-1] < -0.1:
            rrate[-1] = rrate[-2]
        newd = float(deaths[i + 1]) - float(deaths[i])
        drate.append(newd / float(confirmed[i]))
        if drate[-1] > 0.02:
            drate[-1] = drate[-2]

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

    t_range_subdt = [start_date + dt.timedelta(days=x + 1) for x in range(days - 1)]
    plt.figure()
    plt.plot(t_range_subdt, newconf, "b*:", label='Real Dailynew I')
    plt.title('Dailynew I Compare')
    plt.ylabel('number')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t_range_subdt, rrate, "b*:", label='Real Dailynew alpha')
    plt.title('Dailynew alpha Compare')
    plt.ylabel('number')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t_range_subdt, drate, "b*:", label='Real Dailynew gamma')
    plt.title('Dailynew gamma Compare')
    plt.ylabel('number')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t_range_subdt, beta_true, "b*:", label='Real Dailynew u')
    plt.title('Dailynew u Compare')
    plt.ylabel('number')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()
