from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def canBecomeEqual(x: int, y: int) -> bool:
            if x == y:
                return True
            str_x, str_y = str(x), str(y)
            if sorted(str_x) == sorted(str_y):
                return True
            return False
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if canBecomeEqual(nums[i], nums[j]):
                    count += 1
        return count