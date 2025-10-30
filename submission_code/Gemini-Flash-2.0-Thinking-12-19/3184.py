import collections

class SegmentTree:
    def __init__(self, size):
        self.size = size
        self.tree = [-float('inf')] * (4 * size)

    def query(self, node, start, end, left, right):
        if right < start or end < left:
            return -float('inf')
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        p1 = self.query(2 * node, start, mid, left, right)
        p2 = self.query(2 * node + 1, mid + 1, end, left, right)
        return max(p1, p2)

    def update(self, node, start, end, index, val):
        if start == end:
            self.tree[node] = max(self.tree[node], val)
            return
        mid = (start + end) // 2
        if start <= index <= mid:
            self.update(2 * node, start, mid, index, val)
        else:
            self.update(2 * node + 1, mid + 1, end, index, val)
        self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        diffs = [nums[i] - i for i in range(n)]
        unique_diffs = sorted(list(set(diffs)))
        diff_to_rank = {diff: rank + 1 for rank, diff in enumerate(unique_diffs)}
        ranks = [diff_to_rank[diff] for diff in diffs]
        m = len(unique_diffs)
        segment_tree = SegmentTree(m)
        dp = [0] * n
        max_sum = -float('inf')
        for i in range(n):
            rank_i = ranks[i]
            max_prev_dp = segment_tree.query(1, 1, m, 1, rank_i)
            dp[i] = nums[i] + max(0, max_prev_dp)
            segment_tree.update(1, 1, m, rank_i, dp[i])
            max_sum = max(max_sum, dp[i])
        
        if max_sum == -float('inf'):
            return max(nums) if max(nums) <= 0 else max(0, max(nums))
        
        if max_sum < 0:
            return max(nums) if any(x >= 0 for x in nums) else max(nums)
            
        max_result = -float('inf')
        for val in dp:
            max_result = max(max_result, val)
            
        if max_result == -float('inf'):
            return max(nums) if nums else 0

        
        
        result = -float('inf')
        for val in dp:
            result = max(result, val)
            
        if result == -float('inf'):
            return max(nums) if nums else 0
            
        if result < 0:
            max_pos_val = -float('inf')
            for num in nums:
                max_pos_val = max(max_pos_val, num)
            return max(max_pos_val, 0) if max_pos_val != -float('inf') else 0
            
        if result == -float('inf'):
            return max(nums) if nums else 0
            
        final_max_sum = -float('inf')
        for val in dp:
            final_max_sum = max(final_max_sum, val)
            
        if final_max_sum == -float('inf'):
            return max(nums) if nums else 0
            
        if final_max_sum < 0:
            max_val_in_nums = max(nums)
            return max(max_val_in_nums, 0) if max_val_in_nums != -float('inf') else 0
            
        return final_max_sum