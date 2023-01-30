import json

history_length = 5  # 0~4
u_length = 14  # 5~18
target_length = 4  # 18~21

city_size = {'large_city': 10000000., 'normal_city': 1000000., 'small_town': 100000.}
vaccine = {'most_vaccined': .95 * .7, 'normal_vaccined': .7 * .7}
init_epidemic = {'in_epidemic': .03, 'not_epidemic': .001}

init_settings = {}
for city in city_size.keys():
    for vac in vaccine.keys():
        for epi in init_epidemic.keys():
            scenario = city + '-' + vac + '-' + epi
            popu = city_size[city]
            r = popu * vaccine[vac]
            si = popu - r
            i = si * init_epidemic[epi]
            s = si - i
            init_sird = [s, i, r, 0]
            init_settings[scenario] = init_sird

r0_init = {'high_mobility': 3.0, 'normal_mobility': 2.5, 'low_mobility': 2.0}
infect_latency = 2.3
r0_control = {'strict_target': [1.0, 0.8], 'normal_target': [1.0, 0.9], 'loose_target': [1.1, 0.9]}
rr = 0.1
dr = 0.008 / infect_latency

final = {}
for scenario in init_settings.keys():
    for ri in r0_init.keys():
        for rc in r0_control.keys():
            sce_all = scenario + '@' + ri + '-' + rc
            final[sce_all] = {}
            u = [r0_init[ri] / infect_latency]
            for day in range(3):
                u.append(u[-1] * r0_control[rc][0])
            for day in range(17):
                u.append(u[-1] * r0_control[rc][1])

            final[sce_all]['u_his'] = u

            s = init_settings[scenario][0]
            i = init_settings[scenario][1]
            r = init_settings[scenario][2]
            d = init_settings[scenario][3]
            time = 0
            for ut in u:
                si = min(s * i * ut / (s + i + r + d), s)
                ir = i * rr
                id = i * dr
                r = r + ir
                d = d + id
                s = s - si
                i = i + si - ir - id
                time += 1
                if time == 4:
                    final[sce_all]['init_sird'] = [s, i, r, d]
                    final[sce_all]['target'] = []
                if time >= 18:
                    final[sce_all]['target'].append([s, i, r, d])
json.dump(final, open('../eval_settings.json', 'w', encoding='utf8'))
