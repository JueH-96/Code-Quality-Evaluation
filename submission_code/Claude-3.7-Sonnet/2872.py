class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Start from the rightmost element
        current_sum = nums[n-1]
        
        # Iterate from right to left
        for i in range(n-2, -1, -1):
            # If current element can be merged with the right group
            if nums[i] <= current_sum:
                current_sum += nums[i]
            else:
                # Start a new group
                current_sum = nums[i]
        
        return current_sum