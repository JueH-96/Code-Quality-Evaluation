import sys
sys.setrecursionlimit(300000)

class SegmentTreeNode:
    __slots__ = ('total', 'best', 'best_prefix', 'best_suffix')
    def __init__(self, total, best, best_prefix, best_suffix):
        self.total = total
        self.best = best
        self.best_prefix = best_prefix
        self.best_suffix = best_suffix

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.data = data
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = [None] * (2 * self.size)
        self._build(0, 0, self.n-1)
    
    def _build(self, idx, l, r):
        if l == r:
            val = self.data[l]
            self.tree[idx] = SegmentTreeNode(val, val, val, val)
            return
        mid = (l + r) // 2
        self._build(2*idx+1, l, mid)
        self._build(2*idx+2, mid+1, r)
        self.tree[idx] = self._combine(self.tree[2*idx+1], self.tree[2*idx+2])
    
    def _combine(self, left, right):
        if left is None and right is None:
            return None
        if left is None:
            return right
        if right is None:
            return left
        total = left.total + right.total
        best = max(left.best, right.best, left.best_suffix + right.best_prefix)
        best_prefix = max(left.best_prefix, left.total + right.best_prefix)
        best_suffix = max(right.best_suffix, right.total + left.best_suffix)
        return SegmentTreeNode(total, best, best_prefix, best_suffix)
    
    def query(self, l, r):
        return self._query(0, 0, self.n-1, l, r)
    
    def _query(self, idx, segL, segR, l, r):
        if r < segL or l > segR:
            return None
        if l <= segL and segR <= r:
            return self.tree[idx]
        mid = (segL + segR) // 2
        left_res = self._query(2*idx+1, segL, mid, l, r)
        right_res = self._query(2*idx+2, mid+1, segR, l, r)
        if left_res is None:
            return right_res
        if right_res is None:
            return left_res
        return self._combine(left_res, right_res)

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        ans = self.kadane(nums)
        seg_tree = SegmentTree(nums)
        from collections import defaultdict
        index_map = defaultdict(list)
        for i, num in enumerate(nums):
            index_map[num].append(i)
        
        for x, indices in index_map.items():
            indices.sort()
            segments = []
            l = 0
            for i in indices:
                if l <= i-1:
                    segments.append((l, i-1))
                l = i+1
            if l <= n-1:
                segments.append((l, n-1))
            if not segments:
                continue
            best_spanning = -10**18
            best_suffix = -10**18
            for seg in segments:
                node = seg_tree.query(seg[0], seg[1])
                if best_suffix == -10**18:
                    best_spanning = node.best
                    best_suffix = node.best_suffix
                else:
                    best_spanning = max(best_spanning, node.best, best_suffix + node.best_prefix)
                    best_suffix = max(node.best_suffix, best_suffix + node.total)
            best_spanning = max(best_spanning, best_suffix)
            ans = max(ans, best_spanning)
        return ans

    def kadane(self, nums):
        if not nums:
            return 0
        best = nums[0]
        current = nums[0]
        for i in range(1, len(nums)):
            current = max(nums[i], current + nums[i])
            best = max(best, current)
        return best