class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        # Initialize the result as the first element of nums
        result = nums[0]
        
        # Iterate over the rest of the nums
        for i in range(1, len(nums)):
            # Update the result with the bitwise OR of the current element and the next one
            result |= nums[i]
        
        # Return the result after applying k operations
        return result << k