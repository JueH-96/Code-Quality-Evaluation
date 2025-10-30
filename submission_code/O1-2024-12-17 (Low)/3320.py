class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # If there are fewer than 2 elements, no operations can be performed
        if len(nums) < 2:
            return 0
        
        # The sum required for every operation is fixed by the first two elements
        required_sum = nums[0] + nums[1]
        
        count = 0
        i = 0  # Pointer to the current front of the list
        
        # While there are at least 2 elements remaining, check if they match required_sum
        while i + 1 < len(nums):
            if nums[i] + nums[i+1] == required_sum:
                count += 1
                i += 2  # Move past the two used elements
            else:
                break  # As soon as a pair doesn't match, we can't do any more operations
        
        return count