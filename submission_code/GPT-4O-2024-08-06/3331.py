class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Sort the array in non-decreasing order
        nums.sort()
        
        # Initialize the number of operations
        operations = 0
        
        # Iterate over the sorted array
        for num in nums:
            # If the current number is less than k, we need to remove it
            if num < k:
                operations += 1
            else:
                # As soon as we find a number >= k, we can stop
                break
        
        return operations