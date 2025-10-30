import sys
sys.setrecursionlimit(300000)

class _SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(0, 0, self.n-1, data)
    
    def _build(self, idx, tl, tr, data):
        if tl == tr:
            self.tree[idx] = data[tl]
            return
        mid = (tl + tr) // 2
        self._build(2*idx+1, tl, mid, data)
        self._build(2*idx+2, mid+1, tr, data)
        self.tree[idx] = min(self.tree[2*idx+1], self.tree[2*idx+2])
    
    def _push(self, idx, tl, tr):
        if tl == tr:
            self.lazy[idx] = 0
            return
        if self.lazy[idx] != 0:
            left_idx = 2*idx+1
            right_idx = 2*idx+2
            self.tree[left_idx] += self.lazy[idx]
            self.lazy[left_idx] += self.lazy[idx]
            self.tree[right_idx] += self.lazy[idx]
            self.lazy[right_idx] += self.lazy[idx]
            self.lazy[idx] = 0

    def update(self, idx, tl, tr, l, r, delta):
        if r < tl or tr < l:
            return
        if l <= tl and tr <= r:
            self.tree[idx] += delta
            self.lazy[idx] += delta
            return
        self._push(idx, tl, tr)
        mid = (tl + tr) // 2
        self.update(2*idx+1, tl, mid, l, r, delta)
        self.update(2*idx+2, mid+1, tr, l, r, delta)
        self.tree[idx] = min(self.tree[2*idx+1], self.tree[2*idx+2])
    
    def query(self, idx, tl, tr, l, r):
        if r < tl or tr < l:
            return float('inf')
        if l <= tl and tr <= r:
            return self.tree[idx]
        self._push(idx, tl, tr)
        mid = (tl + tr) // 2
        left_min = self.query(2*idx+1, tl, mid, l, r)
        right_min = self.query(2*idx+2, mid+1, tr, l, r)
        return min(left_min, right_min)

class Solution:
    def maxRemoval(self, nums: list, queries: list) -> int:
        n = len(nums)
        diff = [0] * (n+1)
        for l, r in queries:
            diff[l] += 1
            if r+1 < n:
                diff[r+1] -= 1
        A = [0] * n
        A[0] = diff[0]
        for i in range(1, n):
            A[i] = A[i-1] + diff[i]
        
        for i in range(n):
            if A[i] < nums[i]:
                return -1
        
        R = [A[i] - nums[i] for i in range(n)]
        
        seg_tree = _SegmentTree(R)
        
        sorted_queries = sorted(queries, key=lambda q: (q[1], q[0]))
        removed = 0
        for l, r in sorted_queries:
            min_val = seg_tree.query(0, 0, n-1, l, r)
            if min_val > 0:
                removed += 1
                seg_tree.update(0, 0, n-1, l, r, -1)
        return removed