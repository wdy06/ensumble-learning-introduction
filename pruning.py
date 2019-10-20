import numpy as np

import support
import entropy
from zeror import ZeroRule
from linear import Linear
from dtree import DecisionTree


def reducederror(node, x, y):
    # if node is not a leaf
    if isinstance(node, PrunedTree):
        # pruning process
        # get the split of right and left
        feat = x[:, node.feat_index]
        val = node.feat_val
        l, r = node.make_split(feat, val)
        if val is np.inf or len(r) == 0:
            return reducederror(node.left, x, y)
        elif len(l) == 0:
            return reducederror(node.right, x, y)

        # update the branch of right and left
        node.left = reducederror(node.left, x[l], y[l])
        node.right = reducederror(node.right, x[r], y[r])

        # calculate score with train data
        p1 = node.predict(x)
        p2 = node.left.predict(x)
        p3 = node.right.predict(x)
        # if classification
        if y.shape[1] > 1:
            # socre as a number of misclassifications
            ya = y.argmax(axis=1)
            d1 = np.sum(p1.argmax(axis=1) != ya)
            d2 = np.sum(p2.argmax(axis=1) != ya)
            d3 = np.sum(p3.argmax(axis=1) != ya)
        else:
            # score as mean squared error
            d1 = np.mean((p1 - y) ** 2)
            d2 = np.mean((p2 - y) ** 2)
            d3 = np.mean((p3 - y) ** 2)

        if d2 <= d1 or d3 <= d1:  # score is not worse with which left or right
            # return node with better score
            if d2 < d3:
                return node.left
            else:
                return node.right

    # return current node
    return node


def getscore(node, score):
    if isinstance(node, PrunedTree):
        if node.score >= 0 and node.score is not np.inf:
            score.append(node.score)
        getscore(node.left, score)
        getscore(node.right, score)


def criticalscore(node, score_max):
    if type(node) is PrunedTree:
        # pruning process
        # update the branch of right and left
        node.left = criticalscore(node.left, score_max)
        node.right = criticalscore(node.right, score_max)
        # delete node
        leftisleaf = not isinstance(node.left, PrunedTree)
        rightisleaf = not isinstance(node.right, PrunedTree)
        # leave one leaf if both are leaf
        if leftisleaf and rightisleaf:
            return node.left
        # leave branch if which one is leaf
        elif leftisleaf and not rightisleaf:
            return node.right
        elif not leftisleaf and rightisleaf:
            return node.left
        # leave node with better score if both are branch
        elif node.left.score < node.right.score:
            return node.left
        else:
            return node.right

    # return current node
    return node


class PrunedTree(DecisionTree):
    def __init__(self, prunfnc='critical', pruntest=False, splitratio=0.5, critical=0.8,
                 max_depth=5, metric=entropy.gini, leaf=ZeroRule, depth=1):
        super().__init__(max_depth=max_depth, metric=metric, leaf=leaf, depth=depth)
        self.prunfnc = prunfnc
        self.pruntest = pruntest
        self.splitratio = splitratio
        self.critical = critical

    def get_node(self):
        return PrunedTree(prunfnc=self.prunfnc, pruntest=self.pruntest, splitratio=self.splitratio, critical=self.critical,
                          max_depth=self.max_depth, metric=self.metric, leaf=self.leaf, depth=self.depth + 1)

    def fit(self, x, y):
        # if depth=1, root node
        if self.depth == 1 and self.prunfnc is not None:
            # data for pruning
            x_t, y_t = x, y

            if self.pruntest:
                n_test = int(round(len(x) * self.splitratio))
                n_idx = np.random.permutation(len(x))
                tmpx = x[n_idx[n_test:]]
                tmpy = y[n_idx[n_test:]]
                x_t = x[n_idx[:n_test]]
                y_t = y[n_idx[:n_test]]
                x = tmpx
                y = tmpy

        # learn decision tree
        self.left = self.leaf()
        self.right = self.leaf()
        left, right = self.split_tree(x, y)
        if self.depth < self.max_depth:
            self.left = self.get_node()
            self.right = self.get_node()
        if self.depth < self.max_depth or self.prunfnc != 'critical':
            if len(left) > 0:
                self.left.fit(x[left], y[left])
            if len(right) > 0:
                self.right.fit(x[right], y[right])

        # pruning process
        # only whene depth = 1, root node
        if self.depth == 1 and self.prunfnc is not None:
            if self.prunfnc == 'reduce':
                reducederror(self, x_t, y_t)
            elif self.prunfnc == 'critical':
                # get score of metrics function when training
                score = []
                getscore(self, score)
                if len(score) > 0:
                    # calculate max score of branch left
                    i = int(round(len(score) * self.critical))
                    score_max = sorted(score)[min(i, len(score) - 1)]
                    # pruning
                    criticalscore(self, score_max)

                    # learn leaf
                self.fit_leaf(x, y)
        return self

    def fit_leaf(self, x, y):
        feat = x[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)

        # learn only leaf
        if len(l) > 0:
            if isinstance(self.left, PrunedTree):
                self.left.fit_leaf(x[l], y[l])
            else:
                self.left.fit(x[l], y[l])
        if len(r) > 0:
            if isinstance(self.right, PrunedTree):
                self.right.fit_leaf(x[r], y[r])
            else:
                self.right.fit(x[r], y[r])


if __name__ == '__main__':
    import pandas as pd
    np.random.seed(1)
    ps = support.get_base_args()
    ps.add_argument('--depth', '-d', type=int, default=5, help='max tree depth')
    ps.add_argument('--test', '-t', action='store_true',
                    help='test split for pruning')
    ps.add_argument('--pruning', '-p', default='critical',
                    help='pruning algorithm')
    ps.add_argument('--ratio', '-a', type=float, default=0.5,
                    help='test size for pruning')
    ps.add_argument('--critical', '-l', type=float,
                    default=0.8, help='value for critical')
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator,
                     header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        mt = entropy.gini
        lf = ZeroRule
        plf = PrunedTree(prunfnc=args.pruning, pruntest=args.test, splitratio=args.ratio,
                         critical=args.critical, metric=mt, leaf=lf, max_depth=args.depth)
        plf.fit(x, y)
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        mt = entropy.deviation
        lf = linear
        plf = PrunedTree(prunfnc=args.pruning, pruntest=args.test, splitratio=args.ratio,
                         critical=args.critical, metric=mt, leaf=lf, max_depth=args.depth)
        plf.fit(x, y)
        support.report_regressor(plf, x, y, args.crossvalidate)
