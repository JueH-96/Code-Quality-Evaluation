class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_value = 0
        current_sum = nums[n-1]
        
        for i in range(n-2, -1, -1):
            if nums[i] <= current_sum:
                current_sum += nums[i]
            else:
                max_value = max(max_value, current_sum)
                current_sum = nums[i]
        
        max_value = max(max_value, current_sum)
        return max_value