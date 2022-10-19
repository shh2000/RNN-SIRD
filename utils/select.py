from matplotlib import pyplot as plt

df = {'fc2seq': [1.1431, 0.0540],
      'fc2fc': [1.1319, 0.0593],
      'seq2seq': [1.1232, 0.0142],
      'seq2seq_sig': [1.1333, 0.0251],
      'truth': [1.3001, 0.0092],
      'fc': [0.7281, 0.9062],
      'rnn': [2.5806, 0.4679],
      'fc2seq, hp=1e5':[1.1430,0.0120],
      'fc2seq, hp=1e10':[1.1245,0.0005]}

x = []
y = []
for i in range(1, 100):
    x.append(1.48 * (1 - i / 100.0))
    y.append((1.48 - x[-1]) / 20.0)
plt.figure()
plt.xscale('log')
plt.yscale('log')
for key in df.keys():
    plt.scatter(df[key][0], df[key][1], label=key)
plt.plot(x, y, label='Finance Loss')
plt.xlabel('Var Loss')
plt.ylabel('Abs Loss')
plt.title('Finance Loss Result(Without Optimizing hp, hp==1e2)')
plt.legend()
plt.savefig('../../pics/model_select.png')
