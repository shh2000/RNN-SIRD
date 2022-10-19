import scipy


class CaseBetaGen():
    def __init__(self, principal):
        self.principal = principal

    def generate(self, beta_series: list, init_sird: dict, parameters: dict = None):
        if self.principal == 'SIRD':
            if 'deathRate' not in parameters.keys() or 'recoverRate' not in parameters.keys():
                return None
            s = init_sird['s']
            i = init_sird['i']
            r = init_sird['r']
            d = init_sird['d']
            results = {'s': [s], 'i': [i], 'r': [r], 'd': [d]}
            n = s + i + r + d
            for beta in beta_series:
                si = s * i * beta / n
                ir = i * parameters['recoverRate']
                id = i * parameters['deathRate']
                r = r + ir
                d = d + id
                s = s - si
                i = i + si - ir - id
                results['s'].append(s)
                results['i'].append(i)
                results['r'].append(r)
                results['d'].append(d)
            return results


if __name__ == '__main__':
    gen = CaseBetaGen('SIRD')
    parameters = {'recoverRate': 0.03, 'deathRate': 0.008}
    result = gen.generate([.3, .2, .3, .4, .5, .6, .1], {'s': 99950, 'i': 40, 'r': 0, 'd': 0}, parameters)
    print(result)
