from collections import defaultdict

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def is_almost_equal(x, y):
            x_str = str(x)
            y_str = str(y)
            if len(x_str) != len(y_str):
                return False
            diff = []
            for a, b in zip(x_str, y_str):
                if a != b:
                    diff.append((a, b))
                if len(diff) > 2:
                    return False
            if len(diff) == 0:
                return True
            if len(diff) == 2:
                (a1, b1), (a2, b2) = diff
                return a1 == b2 and a2 == b1
            return False
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                if is_almost_equal(nums[i], nums[j]):
                    count += 1
        return count