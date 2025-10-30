class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # Sort the array in ascending order
        nums.sort()
        
        # Initialize the count of operations
        count = 0
        
        # Iterate through the array
        i, j = 0, len(nums) - 1
        while i < j:
            # If the sum of the first two elements is equal to the sum of the last two elements
            if nums[i] + nums[j] == nums[0] + nums[-1]:
                # Increment the count of operations
                count += 1
                
                # Move the pointers inward
                i += 1
                j -= 1
            # If the sum is less than the target sum
            elif nums[i] + nums[j] < nums[0] + nums[-1]:
                # Move the left pointer to the right
                i += 1
            # If the sum is greater than the target sum
            else:
                # Move the right pointer to the left
                j -= 1
        
        return count