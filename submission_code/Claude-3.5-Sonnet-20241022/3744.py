class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        def min_ops_for_query(l: int, r: int) -> int:
            # Create array of numbers from l to r
            nums = list(range(l, r + 1))
            operations = 0
            
            # Continue until all numbers are zero
            while any(nums):
                # Find two non-zero numbers to operate on
                non_zero = [i for i, x in enumerate(nums) if x > 0]
                
                # If only one non-zero number left, it's impossible
                if len(non_zero) == 1:
                    return float('inf')
                
                # Take first two non-zero numbers
                i, j = non_zero[0], non_zero[1]
                
                # Replace with floor division by 4
                nums[i] = nums[i] // 4
                nums[j] = nums[j] // 4
                
                operations += 1
            
            return operations
        
        total_operations = 0
        
        for l, r in queries:
            # For each query, calculate minimum operations
            ops = min_ops_for_query(l, r)
            
            # If any query is impossible, return -1
            if ops == float('inf'):
                return -1
                
            total_operations += ops
            
        return total_operations