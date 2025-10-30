class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        def find_min_operations(num, target):
            # Calculate the minimum operations needed to make num a multiple of target
            if num % target == 0:
                return 0
            else:
                return target - (num % target)
        
        total_operations = 0
        
        for t in target:
            min_operations = float('inf')
            for n in nums:
                operations = find_min_operations(n, t)
                min_operations = min(min_operations, operations)
            total_operations += min_operations
        
        return total_operations