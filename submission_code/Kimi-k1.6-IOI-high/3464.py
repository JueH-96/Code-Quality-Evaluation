class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        prev_pos = nums[0]
        prev_neg = float('-inf')
        for i in range(1, n):
            max_prev = max(prev_pos, prev_neg)
            current_pos = max(prev_neg + nums[i], max_prev + nums[i])
            current_neg = prev_pos - nums[i]
            prev_pos, prev_neg = current_pos, current_neg
        return max(prev_pos, prev_neg)