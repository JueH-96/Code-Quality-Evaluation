from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for p in nums:
            found = False
            for x in range(p):
                if (x | (x + 1)) == p:
                    ans.append(x)
                    found = True
                    break
            if not found:
                ans.append(-1)
        return ans