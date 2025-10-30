class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        if not nums:
            return 0
        prev_pos = nums[0]
        prev_neg = float('-inf')
        max_prev = prev_pos
        
        for num in nums[1:]:
            current_pos = max(prev_neg + num, max_prev + num)
            current_neg = prev_pos - num
            current_max = max(current_pos, current_neg)
            prev_pos, prev_neg, max_prev = current_pos, current_neg, current_max
        
        return max_prev