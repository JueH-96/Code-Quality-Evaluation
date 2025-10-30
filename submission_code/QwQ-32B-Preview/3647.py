class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self.build(arr, 1, 0, self.n - 1)
    
    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build(arr, 2 * node, start, mid)
            self.build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
    
    def _push_down(self, node, start, end):
        if self.lazy[node] != 0:
            self.tree[node] -= self.lazy[node]
            if start != end:
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
            self.lazy[node] = 0
    
    def update(self, node, start, end, l, r, val):
        self._push_down(node, start, end)
        if l > end or r < start:
            return
        if l <= start and end <= r:
            self.lazy[node] += val
            self._push_down(node, start, end)
        else:
            mid = (start + end) // 2
            self.update(2 * node, start, mid, l, r, val)
            self.update(2 * node + 1, mid + 1, end, l, r, val)
            self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
    
    def query(self, node, start, end, l, r):
        self._push_down(node, start, end)
        if l > end or r < start:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        left_max = self.query(2 * node, start, mid, l, r)
        right_max = self.query(2 * node + 1, mid + 1, end, l, r)
        return max(left_max, right_max)

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        # Sort queries by end positions
        queries_sorted = sorted(queries, key=lambda x: x[1])
        
        # Initialize segment tree with nums values
        st = SegmentTree(nums)
        
        q = len(queries)
        selected = 0
        
        for query in queries_sorted:
            l, r = query
            # Check if any index in [l, r] needs more decrements
            if st.query(1, 0, len(nums) - 1, l, r) > 0:
                # Select the query and update the segment tree
                st.update(1, 0, len(nums) - 1, l, r, 1)
                selected += 1
        
        # Check if all indices are covered
        if st.query(1, 0, len(nums) - 1, 0, len(nums) - 1) <= 0:
            return q - selected
        else:
            return -1