class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        max_total = nums[0]
        prev_plus = nums[0]
        prev_minus = -float('inf')
        
        for i in range(1, n):
            current_num = nums[i]
            current_plus = max(prev_plus, prev_minus) + current_num
            current_minus = prev_plus - current_num
            
            current_max = max(current_plus, current_minus)
            if current_max > max_total:
                max_total = current_max
            
            prev_plus, prev_minus = current_plus, current_minus
        
        return max_total