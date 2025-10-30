class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Transform: nums[i] - i
        transformed = [nums[i] - i for i in range(n)]
        
        # Coordinate compression
        sorted_vals = sorted(set(transformed))
        coord_map = {v: i for i, v in enumerate(sorted_vals)}
        
        # Segment tree for range maximum query and point update
        class SegmentTree:
            def __init__(self, size):
                self.size = size
                self.tree = [-float('inf')] * (4 * size)
            
            def update(self, node, start, end, idx, val):
                if start == end:
                    self.tree[node] = max(self.tree[node], val)
                else:
                    mid = (start + end) // 2
                    if idx <= mid:
                        self.update(2 * node, start, mid, idx, val)
                    else:
                        self.update(2 * node + 1, mid + 1, end, idx, val)
                    self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
            
            def query(self, node, start, end, l, r):
                if r < start or end < l:
                    return -float('inf')
                if l <= start and end <= r:
                    return self.tree[node]
                mid = (start + end) // 2
                return max(self.query(2 * node, start, mid, l, r),
                          self.query(2 * node + 1, mid + 1, end, l, r))
            
            def update_point(self, idx, val):
                self.update(1, 0, self.size - 1, idx, val)
            
            def query_range(self, l, r):
                if l > r:
                    return -float('inf')
                return self.query(1, 0, self.size - 1, l, r)
        
        seg_tree = SegmentTree(len(sorted_vals))
        max_sum = max(nums)  # At least one element subsequence
        
        for i in range(n):
            coord = coord_map[transformed[i]]
            
            # Query maximum sum ending at positions with transformed value <= current
            prev_max = seg_tree.query_range(0, coord)
            
            # Current sum is either just this element, or previous max + this element
            current_sum = nums[i]
            if prev_max != -float('inf'):
                current_sum = max(current_sum, prev_max + nums[i])
            
            max_sum = max(max_sum, current_sum)
            
            # Update segment tree
            seg_tree.update_point(coord, current_sum)
        
        return max_sum