from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def is_almost_equal(x, y):
            x_str, y_str = str(x), str(y)
            if len(x_str) != len(y_str):
                return False
            diff = []
            for i in range(len(x_str)):
                if x_str[i] != y_str[i]:
                    diff.append(i)
                if len(diff) > 2:
                    return False
            if len(diff) == 0:
                return True
            if len(diff) == 2:
                i, j = diff
                return x_str[i] == y_str[j] and x_str[j] == y_str[i]
            return False
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if is_almost_equal(nums[i], nums[j]):
                    count += 1
        return count