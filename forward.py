s, i, r, d = 9974., 10., 15., 1.
n = 10000.
rr = 0.3
dr = 0.01
beta = [.5, .55, .58, .62, .7,.72,.7,.69,.42,.3]
for k in range(10):
    si = s * i * beta[k] / n
    ir = i * rr
    id = i * dr
    r = r + ir
    d = d + id
    s = s - si
    i = i + si - ir - id
print(s, i, r, d)
