from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for x in nums:
            res = -1
            for y in range(x):
                if (y | (y + 1)) == x:
                    res = y
                    break
            ans.append(res)
        return ans