import json
import numpy as np

df = json.load(open('../eval_results.json', encoding='utf8'))
result = open('../eval_results.csv', 'w', encoding='utf8')
result.write('city-scale,initial-state,basic-mobility,target-level,NPI_se,NPI_tr,mortality,loss\n')
for scenario in df.keys():
    info = scenario.replace('@', '-').split('-')
    result.write(info[0])
    result.write(',')
    result.write(info[1])
    result.write(' and ')
    result.write(info[2])
    result.write(',')
    result.write(info[3].split('_')[0])
    result.write(',')
    result.write(info[4].split('_')[0])
    result.write(',')

    u = df[scenario]['u']
    loss = df[scenario]['loss']
    x = df[scenario]['ird']

    npi_se = np.max(np.abs(np.diff(u)))
    result.write(str(float(npi_se)))
    result.write(',')

    edge = 0.1 * u[0] + 0.9 * u[-1]
    for i in range(len(u)):
        if u[i] < edge:
            result.write(str(i))
            result.write(',')
            break

    mortality = x[-1][2] / (x[-1][0] + x[-1][1] + x[-1][2])
    result.write(str(mortality))
    result.write(',')

    result.write(str(loss))
    result.write('\n')
