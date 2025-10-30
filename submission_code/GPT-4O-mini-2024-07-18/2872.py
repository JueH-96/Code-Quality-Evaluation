class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        max_value = 0
        current_sum = 0
        
        for i in range(len(nums) - 1, -1, -1):
            if current_sum + nums[i] >= max_value:
                current_sum += nums[i]
                max_value = current_sum
        
        return max_value