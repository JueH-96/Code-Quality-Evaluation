from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        result = []
        for p in nums:
            ans = -1
            for x in range(p):
                if (x | (x + 1)) == p:
                    ans = x
                    break
            result.append(ans)
        return result