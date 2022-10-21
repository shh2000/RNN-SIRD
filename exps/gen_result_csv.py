r = open('../result.csv', 'w', encoding='gbk')
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
r.write(
    'Country,Date,Loss Prediction,Relative Finance Abs Loss(bigger is better because Lfa < 0),Relative Finance Var Loss(less is better)\n')

for area in area2popu.keys():
    for date in dates:
        log_filename = '../dats/' + area + '_' + date + '_loss.log'
        df = open(log_filename)
        lp = float(df.readline().replace('\n', '').split(':')[1])
        fa_pred = float(df.readline().replace('\n', '').split(':')[1])
        fv_pred = float(df.readline().replace('\n', '').split(':')[1])
        fa_true = float(df.readline().replace('\n', '').split(':')[1])
        fv_true = float(df.readline().replace('\n', '').split(':')[1])
        a_score = round((fa_pred / fa_true), 3)
        v_score = round((fv_pred / fv_true), 3)
        str_info = area + ',' + date + ',' + str(lp) + ',' + str(a_score) + ',' + str(v_score) + '\n'
        r.write(str_info)
