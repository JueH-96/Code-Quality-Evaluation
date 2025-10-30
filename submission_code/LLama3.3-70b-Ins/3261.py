from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        res = float('inf')
        
        # Generate all possible combinations of operations
        def backtrack(index, operations, current_nums):
            nonlocal res
            if operations == k or index == n - 1:
                # Calculate the bitwise OR of the remaining elements
                or_result = 0
                for num in current_nums:
                    or_result |= num
                res = min(res, or_result)
                return
            
            # Try to apply an operation at the current index
            if index < n - 1:
                new_nums = current_nums[:index] + [current_nums[index] & current_nums[index + 1]] + current_nums[index + 2:]
                backtrack(index, operations + 1, new_nums)
            
            # Try not to apply an operation at the current index
            backtrack(index + 1, operations, current_nums)
        
        backtrack(0, 0, nums)
        return res