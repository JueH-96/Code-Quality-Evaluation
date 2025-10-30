from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Calculate adjusted values: nums[i] - i
        adjusted = [nums[i] - i for i in range(n)]
        
        # Coordinate compression
        all_vals = sorted(set(adjusted))
        coord_map = {val: i for i, val in enumerate(all_vals)}
        
        # Segment tree for range maximum query
        class SegmentTree:
            def __init__(self, size):
                self.size = size
                self.tree = [-float('inf')] * (4 * size) if size > 0 else []
            
            def update(self, v, tl, tr, pos, val):
                if tl == tr:
                    self.tree[v] = max(self.tree[v], val)
                else:
                    tm = (tl + tr) // 2
                    if pos <= tm:
                        self.update(2*v, tl, tm, pos, val)
                    else:
                        self.update(2*v+1, tm+1, tr, pos, val)
                    self.tree[v] = max(self.tree[2*v], self.tree[2*v+1])
            
            def query(self, v, tl, tr, l, r):
                if l > r:
                    return -float('inf')
                if l == tl and r == tr:
                    return self.tree[v]
                tm = (tl + tr) // 2
                return max(self.query(2*v, tl, tm, l, min(r, tm)),
                          self.query(2*v+1, tm+1, tr, max(l, tm+1), r))
            
            def update_point(self, pos, val):
                if self.size > 0:
                    self.update(1, 0, self.size-1, pos, val)
            
            def query_range(self, l, r):
                if l > r or self.size == 0:
                    return -float('inf')
                return self.query(1, 0, self.size-1, l, r)
        
        seg_tree = SegmentTree(len(all_vals))
        max_sum = max(nums)  # At least one element subsequence
        
        for i in range(n):
            coord = coord_map[adjusted[i]]
            
            # Find maximum sum of valid subsequence ending before position i
            prev_max = seg_tree.query_range(0, coord)
            
            # Current maximum sum ending at position i
            current_sum = nums[i]
            if prev_max > -float('inf'):
                current_sum = max(current_sum, nums[i] + prev_max)
            
            max_sum = max(max_sum, current_sum)
            
            # Update segment tree
            seg_tree.update_point(coord, current_sum)
        
        return max_sum