class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
            
        # Start from right to left
        result = nums[-1]
        current = nums[-1]
        
        for i in range(n-2, -1, -1):
            if nums[i] <= current:
                # If current number is less than or equal to next number
                # We can combine them
                current += nums[i]
            else:
                # If current number is greater, we can't combine
                # Reset current to this number
                current = nums[i]
            result = max(result, current)
            
        return result