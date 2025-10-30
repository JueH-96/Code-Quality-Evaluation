class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Check if already distinct
        if len(set(nums)) == len(nums):
            return 0
        
        operations = 0
        removed = 0
        
        while removed < n:
            operations += 1
            if n - removed <= 3:
                # This operation removes all remaining elements
                return operations
            else:
                # This operation removes exactly 3 elements
                removed += 3
                # Check if remaining elements are distinct
                if len(set(nums[removed:])) == len(nums[removed:]):
                    return operations
        
        return operations