class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        needed = set(range(1, k+1))
        operations = 0
        
        # We remove elements from the end, adding each to our collection
        for i in reversed(range(len(nums))):
            operations += 1
            if nums[i] in needed:
                needed.remove(nums[i])
            if not needed:  # If we've collected all 1..k, we're done
                return operations
        
        return operations