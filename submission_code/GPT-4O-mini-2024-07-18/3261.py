class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        from functools import reduce
        
        # Calculate the initial bitwise OR of the entire array
        initial_or = reduce(lambda x, y: x | y, nums)
        
        # If no operations can be performed, return the initial OR
        if k == 0:
            return initial_or
        
        # We will try to minimize the OR by performing up to k operations
        # We can use a greedy approach to find the minimum possible OR
        n = len(nums)
        
        # We will keep track of the minimum OR we can achieve
        min_or = initial_or
        
        # We can perform operations on adjacent pairs
        for i in range(n - 1):
            # Perform the AND operation on the current pair
            new_value = nums[i] & nums[i + 1]
            # Create a new list simulating the operation
            new_nums = nums[:i] + [new_value] + nums[i + 2:]
            
            # Calculate the OR of the new list
            current_or = reduce(lambda x, y: x | y, new_nums)
            # Update the minimum OR found
            min_or = min(min_or, current_or)
        
        # Return the minimum OR found after considering all pairs
        return min_or