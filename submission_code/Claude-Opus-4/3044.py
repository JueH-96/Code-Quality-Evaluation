class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Set to track which numbers from 1 to k we need to collect
        needed = set(range(1, k + 1))
        operations = 0
        
        # Start from the end of the array
        for i in range(len(nums) - 1, -1, -1):
            operations += 1
            
            # If this number is one we need, remove it from needed set
            if nums[i] in needed:
                needed.remove(nums[i])
            
            # If we've collected all numbers from 1 to k, we're done
            if not needed:
                return operations
        
        return operations