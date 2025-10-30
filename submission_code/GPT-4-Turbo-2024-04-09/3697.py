class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        from math import ceil
        
        def min_operations_for_target(t):
            # Find the minimum operations needed for each target element
            min_operations = float('inf')
            for num in nums:
                if t % num == 0:
                    # If num is already a divisor of t, no operations needed
                    min_operations = 0
                else:
                    # Calculate the minimum increment needed to make num a divisor of t
                    increment = num - (t % num)
                    min_operations = min(min_operations, increment)
            return min_operations
        
        total_operations = 0
        for t in target:
            total_operations += min_operations_for_target(t)
        
        return total_operations