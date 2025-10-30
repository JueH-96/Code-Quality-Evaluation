from typing import List
from collections import defaultdict

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def are_almost_equal(x, y):
            str_x, str_y = str(x), str(y)
            if len(str_x) != len(str_y):
                return False
            diff_count = 0
            for a, b in zip(str_x, str_y):
                if a != b:
                    diff_count += 1
            return diff_count == 2 or diff_count == 0

        n = len(nums)
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                if are_almost_equal(nums[i], nums[j]):
                    count += 1

        return count