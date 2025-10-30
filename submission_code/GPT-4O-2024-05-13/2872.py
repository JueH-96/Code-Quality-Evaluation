class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # We will iterate from the end of the list to the beginning
        # and accumulate the sum when the current element is less than or equal to the next element.
        n = len(nums)
        max_value = nums[-1]
        current_sum = nums[-1]
        
        for i in range(n - 2, -1, -1):
            if nums[i] <= current_sum:
                current_sum += nums[i]
            else:
                current_sum = nums[i]
            max_value = max(max_value, current_sum)
        
        return max_value