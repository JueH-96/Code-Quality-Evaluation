class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Create a set to track collected elements
        collected = set()
        
        # Start from the end of the list
        operations = 0
        for i in range(len(nums) - 1, -1, -1):
            # Add the current element to the collected set
            collected.add(nums[i])
            # Increment the number of operations
            operations += 1
            # Check if we have collected all elements from 1 to k
            if len(collected) == k:
                return operations
        
        # If somehow we exit the loop without collecting all, return the total operations
        return operations