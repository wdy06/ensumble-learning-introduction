import numpy as np
import support


class Linear:
    def __init__(self, epochs=20, lr=0.01, earlystop=None):
        self.epochs = epochs
        self.lr = lr
        self.earlystop = earlystop
        self.beta = None
        self.norm = None

    def __str__(self):
        if type(self.beta) is not type(None):
            s = [str(self.beta[0])]
            e = [f' + feat[{i+1}] * {j}' for i, j in enumerate(self.beta[1:])]
            s.extend(e)
            return ''.join(s)
        else:
            return '0.0'

    def fitnorm(self, x, y):
        self.norm = np.zeros((x.shape[1] + 1, 2))
        self.norm[0, 0] = np.min(y)  # min of objective variable
        self.norm[0, 1] = np.max(y)  # max of objective variable
        self.norm[1:, 0] = np.min(x, axis=0)  # min of explanatory variable
        self.norm[1:, 1] = np.max(x, axis=0)  # max of explanatory variable

    def normalize(self, x, y=None):
        l = self.norm[1:, 1] - self.norm[1:, 0]
        l[l == 0] = 1
        p = (x - self.norm[1:, 0]) / l
        q = y
        if y is not None and not self.norm[0, 1] == self.norm[0, 0]:
            q = (y - self.norm[0, 0]) / (self.norm[0, 1] - self.norm[0, 0])
        return p, q

    def r2(self, y, z):
        # calulate r2 score for earlystop
        y = y.reshape((-1,))
        z = z.reshape((-1,))
        mn = ((y - z) ** 2).sum(axis=0)
        dn = ((y - y.mean()) ** 2).sum(axis=0)
        if dn == 0:
            return np.inf
        return 1.0 - mn / dn

    def fit(self, x, y):
        # estimate linear regression parameters by gradient descent
        self.fitnorm(x, y)
        x, y = self.normalize(x, y)

        # linear regression parameters
        self.beta = np.zeros((x.shape[1] + 1,))

        for _ in range(self.epochs):
            for p, q in zip(x, y):
                z = self.predict(p.reshape((1, -1)), normalize=True)
                z = z.reshape((1,))
                err = (z - q) * self.lr
                delta = p * err
                # update models
                self.beta[0] -= err
                self.beta[1:] -= delta

            if self.earlystop:
                z = self.predict(x, normalize=True)
                s = self.r2(y, z)
                if self.earlystop <= s:
                    break
        return self

    def predict(self, x, normalize=False):
        #print(x)
        if not normalize:
            x, _ = self.normalize(x)

        z = np.zeros((x.shape[0], 1)) + self.beta[0]
        for i in range(x.shape[1]):
            c = x[:, i] * self.beta[i + 1]
            z += c.reshape((-1, 1))

        if not normalize:
            z = z * (self.norm[0, 1] - self.norm[0, 0]) + self.norm[0, 0]
        return z


if __name__ == '__main__':
    import pandas as pd
    ps = support.get_base_args()
    ps.add_argument('--epochs', '-p', type=int,
                    default=20, help='number of epochs')
    ps.add_argument('--learningrate', '-l', type=float,
                    default=0.01, help='learning rate')
    ps.add_argument('--earlystop', '-a', action='store_true',
                    help='Early stopping')
    ps.add_argument('--stoppingvalue', '-v', type=float,
                    default=0.01, help='stopping value')
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator,
                     header=args.header, index_col=args.indexcol, engine='python')
    x = df[df.columns[:-1]].values

    if not args.regression:
        print('Not Support')
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        if args.earlystop:
            plf = Linear(epochs=args.epochs, lr=args.learningrate,
                         earlystop=args.stoppingvalue)
        else:
            plf = Linear(epochs=args.epochs, lr=args.learningrate)

        support.report_regressor(plf, x, y, args.crossvalidate)
