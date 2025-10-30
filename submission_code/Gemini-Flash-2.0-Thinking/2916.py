from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n <= 1:
            return True

        memo = {}

        def can_split(arr_tuple):
            if arr_tuple in memo:
                return memo[arr_tuple]

            arr = list(arr_tuple)
            if len(arr) == 1:
                return True

            for i in range(1, len(arr)):
                left = arr[:i]
                right = arr[i:]

                is_left_ok = len(left) == 1 or sum(left) >= m
                is_right_ok = len(right) == 1 or sum(right) >= m

                if is_left_ok and is_right_ok:
                    if can_split(tuple(left)) and can_split(tuple(right)):
                        memo[arr_tuple] = True
                        return True

            memo[arr_tuple] = False
            return False

        return can_split(tuple(nums))