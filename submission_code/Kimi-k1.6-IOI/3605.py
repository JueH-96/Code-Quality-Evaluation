from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for m in nums:
            found = False
            for x in range(m):
                if (x | (x + 1)) == m:
                    ans.append(x)
                    found = True
                    break
            if not found:
                ans.append(-1)
        return ans