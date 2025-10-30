class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        def is_distinct(arr):
            return len(arr) == len(set(arr))
        
        ops = 0
        i = 0
        n = len(nums)
        
        while True:
            # Check if the current subarray is already distinct
            if is_distinct(nums[i:]):
                return ops
            
            # If at least 3 elements remain, remove the first 3
            if n - i >= 3:
                i += 3
                ops += 1
            else:
                # Remove all remaining (fewer than 3) elements
                ops += 1
                return ops