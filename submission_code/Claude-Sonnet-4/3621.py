class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If any element is less than k, impossible
        if any(num < k for num in nums):
            return -1
        
        # Get all unique values greater than k
        unique_vals = sorted(set(nums))
        
        # If k is not in the array, we need to be able to reach it
        if k not in unique_vals:
            unique_vals.append(k)
            unique_vals.sort()
        
        # Count operations: we need one operation for each unique value > k
        operations = 0
        for val in unique_vals:
            if val > k:
                operations += 1
        
        return operations