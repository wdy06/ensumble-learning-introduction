import numpy as np

import support
import entropy
from zeror import ZeroRule
from linear import Linear


class DecisionStump:
    def __init__(self, metric=entropy.gini, leaf=ZeroRule):
        self.metric = metric # metric function
        self.leaf = leaf # leaf model
        self.left = None
        self.right = None
        self.feat_index = 0 # objective variable index to split
        self.feat_val = np.nan # objective variable value to split
        self.score = np.nan # score caluclated by metric function

    def make_split(self, feat, val):
        # return feat index splitted by val
        left, right = [], []
        for i, v in enumerate(feat):
            if v < val:
                left.append(i)
            else:
                right.append(i)
        return left, right

    def make_loss(self, y1, y2):
        y1_size = y1.shape[0]
        y2_size = y2.shape[0]
        if y1_size == 0 or y2_size == 0:
            return np.inf
        totol = y1_size + y2_size
        m1 = self.metric(y1) * (y1_size / totol)
        m2 = self.metric(y2) * (y2_size / totol)
        return m1 + m2

    def split_tree(self,x, y):
        # split data and return index belonging right and left
        self.feat_index = 0
        self.feat_val = np.inf
        score = np.inf
        left, right = list(range(x.shape[0])), []
        # to all dimensions of explanatory variable
        for i in range(x.shape[1]):
            feat = x[:, i]
            for val in feat:
                # find value to split best
                l, r = self.make_split(feat, val)
                loss = self.make_loss(y[l], y[r])
                if score > loss:
                    score = loss
                    left = l
                    right = r
                    self.feat_index = i
                    self.feat_val = val
        self.score = score
        return left, right

    def fit(self, x, y):
        # create leaf of left and right
        self.left = self.leaf()
        self.right = self.leaf()

        # distribute data to left and right
        left, right = self.split_tree(x, y)
        # learn leaf model of left and right
        if len(left) > 0:
            self.left.fit(x[left], y[left])
        if len(right) > 0:
            self.right.fit(x[right], y[right])
        return self

    def predict(self, x):
        feat = x[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)
        z = None
        if len(l) > 0 and len(r) > 0:
            left = self.left.predict(x[l])
            right = self.right.predict(x[r])
            z = np.zeros((x.shape[0], left.shape[1]))
            z[l] = left
            z[r] = right
        elif len(l) > 0:
            z = self.left.predict(x)
        elif len(r) > 0:
            z = self.right.predict(x)
        return z

    def __str__(self):
        return '\n'.join([
            f'if feat[{self.feat_index}] <= {self.feat_val}'
            f'    {self.left}',
            'else',
            f'    {self.right}'
        ])

if __name__ == '__main__':
    import pandas as pd
    ps = support.get_base_args()
    ps.add_argument('--metric', '-m', default='', help='metric function')
    ps.add_argument('--leaf', '-l', default='', help='leaf model')
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if args.metric == 'div':
        mt = entropy.deviation
    elif args.metric == 'infgain':
        mt = entropy.infgain
    elif args.metric == 'gini':
        mt = entropy.gini
    else:
        mt = None

    if args.leaf == 'zeror':
        lf = ZeroRule
    elif args.leaf == 'linear':
        lf = Linear
    else:
        lf = None

    if mt is None:
        mt = entropy.gini
    if lf is None:
        lf = ZeroRule
    plf = DecisionStump(metric=mt, leaf=lf)

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        support.report_regressor(plf, x, y, args.crossvalidate)

