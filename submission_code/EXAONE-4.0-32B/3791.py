import sys
sys.setrecursionlimit(300000)

class Solution:
    def __init__(self):
        self.neg_inf = -10**18
        self.tree = []
        self.baskets = []
        self.n = 0

    def numOfUnplacedFruits(self, fruits, baskets):
        self.n = n = len(fruits)
        self.baskets = baskets
        self.tree = [0] * (4 * n)
        self.build(0, 0, n-1)
        unplaced = 0
        for f in fruits:
            idx = self.query_iter(f)
            if idx is None:
                unplaced += 1
            else:
                self.update(0, 0, n-1, idx, self.neg_inf)
        return unplaced

    def build(self, node, l, r):
        if l == r:
            self.tree[node] = self.baskets[l]
            return
        mid = (l + r) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        self.build(left_child, l, mid)
        self.build(right_child, mid+1, r)
        self.tree[node] = max(self.tree[left_child], self.tree[right_child])

    def update(self, node, l, r, idx, val):
        if l == r:
            self.tree[node] = val
            return
        mid = (l + r) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        if idx <= mid:
            self.update(left_child, l, mid, idx, val)
        else:
            self.update(right_child, mid+1, r, idx, val)
        self.tree[node] = max(self.tree[left_child], self.tree[right_child])

    def query_iter(self, f):
        node = 0
        l, r = 0, self.n-1
        while l < r:
            if self.tree[node] < f:
                return None
            mid = (l + r) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            if self.tree[left_child] >= f:
                node = left_child
                r = mid
            else:
                node = right_child
                l = mid + 1
        if self.tree[node] >= f:
            return l
        else:
            return None