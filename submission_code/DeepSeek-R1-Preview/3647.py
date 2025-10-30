import bisect

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.sum_tree = [0] * (2 * self.size)
        self.lazy = [0] * (2 * self.size)
        for i in range(self.n):
            self.sum_tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            self.sum_tree[i] = self.sum_tree[2 * i] + self.sum_tree[2 * i + 1]
    
    def push(self, node, l, r):
        if self.lazy[node] != 0:
            self.sum_tree[node] = max(self.sum_tree[node] - self.lazy[node], 0)
            if l != r:
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
            self.lazy[node] = 0
    
    def range_update(self, a, b, node=1, l=0, r=None):
        if r is None:
            r = self.size - 1
        self.push(node, l, r)
        if a > r or b < l:
            return
        if a <= l and r <= b:
            self.sum_tree[node] = max(self.sum_tree[node] - 1, 0)
            if l != r:
                self.lazy[2 * node] += 1
                self.lazy[2 * node + 1] += 1
            return
        mid = (l + r) // 2
        self.range_update(a, b, 2 * node, l, mid)
        self.range_update(a, b, 2 * node + 1, mid + 1, r)
        self.sum_tree[node] = self.sum_tree[2 * node] + self.sum_tree[2 * node + 1]
    
    def range_query(self, a, b, node=1, l=0, r=None):
        if r is None:
            r = self.size - 1
        self.push(node, l, r)
        if a > r or b < l:
            return 0
        if a <= l and r <= b:
            return self.sum_tree[node]
        mid = (l + r) // 2
        return self.range_query(a, b, 2 * node, l, mid) + self.range_query(a, b, 2 * node + 1, mid + 1, r)

class Solution:
    def maxRemoval(self, nums, queries):
        n = len(nums)
        m = len(queries)
        
        coverage = [0] * n
        for l, r in queries:
            coverage[l] += 1
            if r + 1 < n:
                coverage[r + 1] -= 1
        for i in range(1, n):
            coverage[i] += coverage[i - 1]
        for i in range(n):
            if coverage[i] < nums[i]:
                return -1
        
        st = SegmentTree(nums.copy())
        query_contrib = []
        for i, (l, r) in enumerate(queries):
            contrib = st.range_query(l, r)
            query_contrib.append((-contrib, l, r, i))
        query_contrib.sort()
        
        selected = 0
        for q in query_contrib:
            _, l, r, idx = q
            current_sum = st.range_query(l, r)
            if current_sum > 0:
                st.range_update(l, r)
                selected += 1
        
        return m - selected