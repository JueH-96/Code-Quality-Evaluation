class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        start = 0  # Starting index of the current subarray
        
        while start < len(nums):
            # Check if the current subarray has distinct elements
            # This is true if the number of unique elements equals the subarray length
            if len(set(nums[start:])) == len(nums) - start:
                return operations
            
            # Perform one operation - move the starting index by 3
            start += 3
            operations += 1
        
        return operations