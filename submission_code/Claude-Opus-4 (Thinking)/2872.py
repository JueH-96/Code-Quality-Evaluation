class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        result = 0
        current_sum = 0
        
        # Process from right to left
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] <= current_sum:
                # Can merge with accumulated sum on the right
                current_sum += nums[i]
            else:
                # Can't merge, start new accumulation
                current_sum = nums[i]
            result = max(result, current_sum)
        
        return result