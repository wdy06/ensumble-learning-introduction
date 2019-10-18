import numpy as np
import support
import entropy
from zeror import ZeroRule
from linear import Linear
from dstump import DecisionStump


class DecisionTree(DecisionStump):
    def __init__(self, max_depth=5, metric=entropy.gini, leaf=ZeroRule, depth=1):
        super().__init__(metric=metric, leaf=leaf)
        self.max_depth = max_depth
        self.depth = depth

    def fit(self, x, y):
        # create leaf node of left and right
        self.left = self.leaf()
        self.right = self.leaf()
        # split data into left and right node
        left, right = self.split_tree(x, y)

        if self.depth < self.max_depth:
            if len(left) > 0:
                self.left = self.get_node()
            if len(right) > 0:
                self.right = self.get_node()

        # learn left and right node
        if len(left) > 0:
            self.left.fit(x[left], y[left])
        if len(right) > 0:
            self.right.fit(x[right], y[right])
        return self

    def get_node(self):
        return DecisionTree(max_depth=self.max_depth, metric=self.metric,
                            leaf=self.leaf, depth=self.depth + 1)

    def print_leaf(self, node, d=0):
        if isinstance(node, DecisionTree):
            return '\n'.join([f'{"+"*d}if feat[{d}] <= {node.feat_index} then {node.feat_val}',
                              self.print_leaf(node.left, d+1),
                              f'   {"|"*d}else',
                              self.print_leaf(node.right, d+1)])
        else:
            return f'   {"|"*(d-1)} {node}'

    def __str__(self):
        return self.print_leaf(self)


if __name__ == '__main__':
    import pandas as pd
    ps = support.get_base_args()
    ps.add_argument('--metric', '-m', default='', help='Metric function')
    ps.add_argument('--leaf', '-l', default='', help='Leaf Class')
    ps.add_argument('--depth', '-d', type=int,
                    default=5, help='Max Tree Depth')
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator,
                     header=args.header, index_col=args.indexcol)
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

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        if mt is None:
            mt = entropy.gini
        if lf is None:
            lf = ZeroRule
        plf = DecisionTree(metric=mt, leaf=lf, max_depth=args.depth)
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        if mt is None:
            mt = entropy.deviation
        if lf is None:
            lf = Linear
        plf = DecisionTree(metric=mt, leaf=lf, max_depth=args.depth)
        plf.fit(x, y)
        support.report_regressor(plf, x, y, args.crossvalidate)
