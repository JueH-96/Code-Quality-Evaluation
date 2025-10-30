from bisect import bisect_left, bisect_right
from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        if x == 0:
            # Handle the case separately as any pair is valid
            nums_sorted = sorted(nums)
            min_diff = float('inf')
            for i in range(1, n):
                min_diff = min(min_diff, nums_sorted[i] - nums_sorted[i-1])
            return min_diff
        
        # Coordinate compression
        sorted_vals = sorted(set(nums))
        m = len(sorted_vals)
        if m == 0:
            return 0  # This case is impossible as per problem constraints (n >=1)
        value_to_idx = {v: i for i, v in enumerate(sorted_vals)}
        
        class SegmentTree:
            def __init__(self, size):
                self.size_tree = 1
                while self.size_tree < size:
                    self.size_tree <<= 1
                self.min_tree = [float('inf')] * (2 * self.size_tree)
                self.max_tree = [-float('inf')] * (2 * self.size_tree)
            
            def update(self, pos, value):
                pos += self.size_tree
                self.min_tree[pos] = value
                self.max_tree[pos] = value
                pos >>= 1
                while pos >= 1:
                    left = 2 * pos
                    right = 2 * pos + 1
                    self.min_tree[pos] = min(self.min_tree[left], self.min_tree[right])
                    self.max_tree[pos] = max(self.max_tree[left], self.max_tree[right])
                    pos >>= 1
            
            def query_max(self, l, r):
                res = -float('inf')
                l += self.size_tree
                r += self.size_tree
                while l <= r:
                    if l % 2 == 1:
                        res = max(res, self.max_tree[l])
                        l += 1
                    if r % 2 == 0:
                        res = max(res, self.max_tree[r])
                        r -= 1
                    l >>= 1
                    r >>= 1
                return res
            
            def query_min(self, l, r):
                res = float('inf')
                l += self.size_tree
                r += self.size_tree
                while l <= r:
                    if l % 2 == 1:
                        res = min(res, self.min_tree[l])
                        l += 1
                    if r % 2 == 0:
                        res = min(res, self.min_tree[r])
                        r -= 1
                    l >>= 1
                    r >>= 1
                return res
        
        min_diff = float('inf')
        
        # Left to right pass
        if x <= n - 1:
            st = SegmentTree(m)
            for i in range(x, n):
                val = nums[i - x]
                idx = value_to_idx[val]
                st.update(idx, val)
                current = nums[i]
                # Predecessor
                pos = bisect_right(sorted_vals, current) - 1
                if pos >= 0:
                    max_val = st.query_max(0, pos)
                    if max_val != -float('inf'):
                        min_diff = min(min_diff, current - max_val)
                # Successor
                pos = bisect_left(sorted_vals, current)
                if pos < m:
                    min_val = st.query_min(pos, m - 1)
                    if min_val != float('inf'):
                        min_diff = min(min_diff, min_val - current)
        
        # Right to left pass
        if x <= n - 1:
            st = SegmentTree(m)
            start = n - 1 - x
            for i in range(start, -1, -1):
                j = i + x
                val = nums[j]
                idx = value_to_idx[val]
                st.update(idx, val)
                current = nums[i]
                # Predecessor
                pos = bisect_right(sorted_vals, current) - 1
                if pos >= 0:
                    max_val = st.query_max(0, pos)
                    if max_val != -float('inf'):
                        min_diff = min(min_diff, current - max_val)
                # Successor
                pos = bisect_left(sorted_vals, current)
                if pos < m:
                    min_val = st.query_min(pos, m - 1)
                    if min_val != float('inf'):
                        min_diff = min(min_diff, min_val - current)
        
        return min_diff if min_diff != float('inf') else 0