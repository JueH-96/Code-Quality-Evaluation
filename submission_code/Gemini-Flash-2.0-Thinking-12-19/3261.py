from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        memo = {}
        
        def get_bitwise_or(arr):
            if not arr:
                return 0
            result = 0
            for x in arr:
                result |= x
            return result
            
        def solve(current_nums_tuple, operations_left):
            if operations_left < 0:
                return float('inf')
            if operations_left == 0:
                return get_bitwise_or(current_nums_tuple)
            if len(current_nums_tuple) <= 1:
                return get_bitwise_or(current_nums_tuple)
            if (current_nums_tuple, operations_left) in memo:
                return memo[(current_nums_tuple, operations_left)]
            
            min_or_val = float('inf')
            for i in range(len(current_nums_tuple) - 1):
                next_nums_list = list(current_nums_tuple)
                val = next_nums_list[i] & next_nums_list[i+1]
                next_nums_list = next_nums_list[:i] + [val] + next_nums_list[i+2:]
                next_nums_tuple = tuple(next_nums_list)
                or_val = solve(next_nums_tuple, operations_left - 1)
                min_or_val = min(min_or_val, or_val)
                
            memo[(current_nums_tuple, operations_left)] = min_or_val
            return min_or_val
            
        initial_or = get_bitwise_or(nums)
        min_or_result = solve(tuple(nums), k)
        return min(initial_or, min_or_result)