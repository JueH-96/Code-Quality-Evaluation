from typing import List

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        total_operations = 0
        
        for l, r in queries:
            # Create the initial array from l to r
            nums = list(range(l, r + 1))
            operations = 0
            
            # While there are elements greater than 0
            while any(num > 0 for num in nums):
                # Sort the array to always take the largest two elements
                nums.sort()
                
                # Take the last two elements (largest)
                a = nums[-1]
                b = nums[-2] if len(nums) > 1 else 0
                
                # Replace them with floor(a / 4) and floor(b / 4)
                if a > 0:
                    nums[-1] = a // 4
                if b > 0:
                    nums[-2] = b // 4
                
                operations += 1
            
            total_operations += operations
        
        return total_operations