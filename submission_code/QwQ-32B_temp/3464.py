class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        prev_even = float('-inf')
        prev_odd = nums[0]
        
        for i in range(1, n):
            current_even = prev_odd - nums[i]
            current_odd_candidate1 = prev_even + nums[i]
            current_odd_candidate2 = (max(prev_even, prev_odd)) + nums[i]
            current_odd = max(current_odd_candidate1, current_odd_candidate2)
            
            prev_even, prev_odd = current_even, current_odd
        
        return max(prev_even, prev_odd)