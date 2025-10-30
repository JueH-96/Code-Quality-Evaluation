import sys

# Increase recursion limit for deep segment tree traversals and use fast I/O
sys.setrecursionlimit(2 * 10**5 + 5)

class Node:
    """A node in the segment tree."""
    def __init__(self):
        self.sum = 0
        self.min_val = 0
        self.max_val = 0
        self.lazy_set = None

class SegTree:
    """A lazy segment tree for range sum, range set, and specialized find queries."""
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [Node() for _ in range(4 * self.n)]
        self._build(1, 0, self.n - 1, arr)

    def _build(self, v, tl, tr, arr):
        if tl == tr:
            val = arr[tl]
            self.tree[v].sum = val
            self.tree[v].min_val = val
            self.tree[v].max_val = val
        else:
            tm = (tl + tr) // 2
            self._build(2 * v, tl, tm, arr)
            self._build(2 * v + 1, tm + 1, tr, arr)
            self._pull(v)

    def _pull(self, v):
        left, right = self.tree[2 * v], self.tree[2 * v + 1]
        self.tree[v].sum = left.sum + right.sum
        self.tree[v].min_val = min(left.min_val, right.min_val)
        self.tree[v].max_val = max(left.max_val, right.max_val)

    def _push(self, v, tl, tr):
        if self.tree[v].lazy_set is not None:
            val = self.tree[v].lazy_set
            tm = (tl + tr) // 2
            
            # Apply to left child
            self.tree[2 * v].lazy_set = val
            self.tree[2 * v].sum = val * (tm - tl + 1)
            self.tree[2 * v].min_val = val
            self.tree[2 * v].max_val = val
            
            # Apply to right child
            self.tree[2 * v + 1].lazy_set = val
            self.tree[2 * v + 1].sum = val * (tr - tm)
            self.tree[2 * v + 1].min_val = val
            self.tree[2 * v + 1].max_val = val
            
            self.tree[v].lazy_set = None

    def _query_sum(self, v, tl, tr, l, r):
        if l > r:
            return 0
        if l == tl and r == tr:
            return self.tree[v].sum
        self._push(v, tl, tr)
        tm = (tl + tr) // 2
        left_sum = self._query_sum(2 * v, tl, tm, l, min(r, tm))
        right_sum = self._query_sum(2 * v + 1, tm + 1, tr, max(l, tm + 1), r)
        return left_sum + right_sum

    def query_sum(self, l, r):
        return self._query_sum(1, 0, self.n - 1, l, r)

    def _range_set(self, v, tl, tr, l, r, val):
        if l > r:
            return
        if l == tl and r == tr:
            self.tree[v].lazy_set = val
            self.tree[v].sum = val * (tr - tl + 1)
            self.tree[v].min_val = val
            self.tree[v].max_val = val
            return
        self._push(v, tl, tr)
        tm = (tl + tr) // 2
        self._range_set(2 * v, tl, tm, l, min(r, tm), val)
        self._range_set(2 * v + 1, tm + 1, tr, max(l, tm + 1), r, val)
        self._pull(v)

    def range_set(self, l, r, val):
        self._range_set(1, 0, self.n - 1, l, r, val)

    def _find_first_gt(self, v, tl, tr, l, r, val):
        if l > r or tl > r or tr < l or self.tree[v].max_val <= val:
            return -1
        if tl == tr:
            return tl
        self._push(v, tl, tr)
        tm = (tl + tr) // 2
        res = self._find_first_gt(2 * v, tl, tm, l, r, val)
        if res != -1:
            return res
        return self._find_first_gt(2 * v + 1, tm + 1, tr, l, r, val)

    def find_first_gt(self, l, r, val):
        return self._find_first_gt(1, 0, self.n - 1, l, r, val)

    def _find_last_lt(self, v, tl, tr, l, r, val):
        if l > r or tl > r or tr < l or self.tree[v].min_val >= val:
            return -1
        if tl == tr:
            return tl
        self._push(v, tl, tr)
        tm = (tl + tr) // 2
        res = self._find_last_lt(2 * v + 1, tm + 1, tr, l, r, val)
        if res != -1:
            return res
        return self._find_last_lt(2 * v, tl, tm, l, r, val)
    
    def find_last_lt(self, l, r, val):
        return self._find_last_lt(1, 0, self.n - 1, l, r, val)

def solve():
    N = int(sys.stdin.readline())
    X = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())
    
    c = [X[i] - i for i in range(N)]
    
    seg_tree = SegTree(c)
    
    total_cost = 0
    
    for _ in range(Q):
        T, G = map(int, sys.stdin.readline().split())
        T_idx = T - 1
        C = G - T_idx
        
        cost = 0
        
        # Part 1: Update persons to the left (j < T_idx)
        if T_idx > 0:
            k1 = seg_tree.find_first_gt(0, T_idx - 1, C)
            if k1 != -1:
                range_end = T_idx - 1
                sum1 = seg_tree.query_sum(k1, range_end)
                count1 = range_end - k1 + 1
                cost += sum1 - count1 * C
                seg_tree.range_set(k1, range_end, C)
                
        # Part 2: Update the target person (j = T_idx)
        old_cT = seg_tree.query_sum(T_idx, T_idx)
        cost += abs(C - old_cT)
        seg_tree.range_set(T_idx, T_idx, C)
        
        # Part 3: Update persons to the right (j > T_idx)
        if T_idx < N - 1:
            k2 = seg_tree.find_last_lt(T_idx + 1, N - 1, C)
            if k2 != -1:
                range_start = T_idx + 1
                sum2 = seg_tree.query_sum(range_start, k2)
                count2 = k2 - range_start + 1
                cost += count2 * C - sum2
                seg_tree.range_set(range_start, k2, C)

        total_cost += cost

    print(total_cost)

solve()