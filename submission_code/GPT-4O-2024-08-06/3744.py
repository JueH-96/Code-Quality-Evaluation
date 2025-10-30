class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        def operations_for_range(l, r):
            # Calculate the number of operations needed for a range [l, r]
            ops = 0
            while l <= r:
                # Count how many numbers are in the current range
                count = r - l + 1
                # Calculate the number of operations needed for this range
                ops += count // 2
                # Move to the next range by dividing l and r by 4
                l = l // 4
                r = r // 4
            return ops
        
        total_operations = 0
        for l, r in queries:
            total_operations += operations_for_range(l, r)
        
        return total_operations