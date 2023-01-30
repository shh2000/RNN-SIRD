import numpy as np
import matplotlib.pyplot as plt

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
"""
result = np.ones(shape=(480*11, 640*6, 4))
j = 0
for area in area2popu.keys():
    i = 0
    for date in dates:
        pic_filename = '../pics/' + area + '_' + date + '_whole.png'
        print(result.shape, 480 * j, 480 * j + 480, 640 * i, 640 * i + 640)
        result[480 * j:480 * j + 480, 640 * i:640 * i + 640] = plt.imread(pic_filename)
        i += 1
    j += 1
plt.imshow(result)
plt.imsave('../x_result.png', result)

result = np.ones(shape=(480*11, 640*6, 4))
j = 0
for area in area2popu.keys():
    i = 0
    for date in dates:
        pic_filename = '../pics/' + area + '_' + date + '_u.png'
        print(result.shape, 480 * j, 480 * j + 480, 640 * i, 640 * i + 640)
        result[480 * j:480 * j + 480, 640 * i:640 * i + 640] = plt.imread(pic_filename)
        i += 1
    j += 1
plt.imshow(result)
plt.imsave('../u_result.png', result)
"""
result = np.ones(shape=(480*11, 640*6, 4))
j = 0
for area in area2popu.keys():
    i = 0
    for date in dates:
        pic_filename = '../pics/' + area + '_' + date + '_u_delta.png'
        print(result.shape, 480 * j, 480 * j + 480, 640 * i, 640 * i + 640)
        result[480 * j:480 * j + 480, 640 * i:640 * i + 640] = plt.imread(pic_filename)
        i += 1
    j += 1
plt.imshow(result)
plt.imsave('../u_delta_result.png', result)