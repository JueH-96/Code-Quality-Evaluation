from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        # directions: -1 for left, +1 for right
        for start in range(n):
            if nums[start] != 0:
                continue
            for init_dir in (-1, 1):
                arr = nums[:]         # make a working copy
                curr = start
                d = init_dir
                # simulate until we step out of bounds
                while 0 <= curr < n:
                    if arr[curr] == 0:
                        # just move on
                        curr += d
                    else:
                        # decrement, reverse, then move
                        arr[curr] -= 1
                        d = -d
                        curr += d
                # check if all zeros
                if all(x == 0 for x in arr):
                    res += 1
        return res