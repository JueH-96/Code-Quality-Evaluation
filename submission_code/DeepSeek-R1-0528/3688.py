import sys
sys.setrecursionlimit(300000)

class Node:
    def __init__(self, t, mp, ms, msa):
        self.total = t
        self.max_prefix = mp
        self.max_suffix = ms
        self.max_subarray = msa

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.data = data
        self.tree = [None] * (4 * self.n)
        self._build(0, 0, self.n-1)
    
    def _build(self, idx, l, r):
        if l == r:
            val = self.data[l]
            self.tree[idx] = Node(val, val, val, val)
            return
        mid = (l + r) // 2
        self._build(2*idx+1, l, mid)
        self._build(2*idx+2, mid+1, r)
        self.tree[idx] = self._merge_nodes(self.tree[2*idx+1], self.tree[2*idx+2])
    
    def _merge_nodes(self, left, right):
        total = left.total + right.total
        max_prefix = max(left.max_prefix, left.total + right.max_prefix)
        max_suffix = max(right.max_suffix, right.total + left.max_suffix)
        max_subarray = max(left.max_subarray, right.max_subarray, left.max_suffix + right.max_prefix)
        return Node(total, max_prefix, max_suffix, max_subarray)
    
    def update(self, idx, l, r, pos, val):
        if l == r:
            self.tree[idx] = Node(val, val, val, val)
            return
        mid = (l + r) // 2
        if pos <= mid:
            self.update(2*idx+1, l, mid, pos, val)
        else:
            self.update(2*idx+2, mid+1, r, pos, val)
        self.tree[idx] = self._merge_nodes(self.tree[2*idx+1], self.tree[2*idx+2])
    
    def update_point(self, pos, val):
        self.update(0, 0, self.n-1, pos, val)

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        seg = SegmentTree(nums)
        ans = seg.tree[0].max_subarray
        
        from collections import defaultdict
        neg_map = defaultdict(list)
        for i, num in enumerate(nums):
            if num < 0:
                neg_map[num].append(i)
        
        for x, indices in neg_map.items():
            if len(indices) == n:
                continue
            for i in indices:
                seg.update_point(i, 0)
            candidate = seg.tree[0].max_subarray
            ans = max(ans, candidate)
            for i in indices:
                seg.update_point(i, nums[i])
        
        return ans