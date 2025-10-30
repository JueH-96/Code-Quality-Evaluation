class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if k is present in nums
        if k not in nums:
            return -1
        
        # Sort the nums to easily find the valid integers
        nums.sort()
        
        operations = 0
        current_max = max(nums)
        
        # Start from the largest number and work downwards
        for num in reversed(nums):
            if num > k:
                # We can only set num to a valid h which is less than or equal to current_max
                if current_max > k:
                    operations += 1
                    current_max = num  # Set the current max to the current number
                else:
                    return -1  # If we can't set it to k, return -1
            elif num < k:
                # If we encounter a number less than k, we can't make it equal to k
                return -1
        
        return operations