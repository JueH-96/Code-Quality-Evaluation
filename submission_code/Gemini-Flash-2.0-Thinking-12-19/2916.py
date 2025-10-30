class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n <= 1:
            return True
        
        memo = {}
        
        def solve(arr):
            arr_tuple = tuple(arr)
            if arr_tuple in memo:
                return memo[arr_tuple]
            if len(arr) == 1:
                return True
            for i in range(1, len(arr)):
                left_arr = arr[:i]
                right_arr = arr[i:]
                is_left_valid = (len(left_arr) == 1) or (sum(left_arr) >= m)
                is_right_valid = (len(right_arr) == 1) or (sum(right_arr) >= m)
                if is_left_valid and is_right_valid:
                    if solve(left_arr) and solve(right_arr):
                        memo[arr_tuple] = True
                        return True
            memo[arr_tuple] = False
            return False
            
        return solve(nums)

from typing import List