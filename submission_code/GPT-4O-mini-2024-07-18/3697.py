from typing import List

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        # Create a set of nums for quick lookup
        nums_set = set(nums)
        operations = 0
        
        for t in target:
            # Check if there's already a multiple of t in nums
            if any((t % n == 0) for n in nums):
                continue
            
            # If not, find the minimum operations needed to make a multiple of t
            min_operations = float('inf')
            for n in nums:
                # Calculate how much we need to increment n to be a multiple of t
                if t % n != 0:
                    increment = t - (t // n) * n + (n if t % n != 0 else 0)
                    min_operations = min(min_operations, increment)
            
            operations += min_operations
        
        return operations