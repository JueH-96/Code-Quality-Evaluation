class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Create a set of numbers we need to collect (1 to k)
        needed = set(range(1, k + 1))
        operations = 0
        
        # Process elements from the end of the array
        for num in reversed(nums):
            operations += 1
            
            # If the current number is one we need, remove it from our needed set
            if num in needed:
                needed.remove(num)
            
            # If we've collected all numbers from 1 to k, return the operation count
            if not needed:
                return operations
        
        return operations