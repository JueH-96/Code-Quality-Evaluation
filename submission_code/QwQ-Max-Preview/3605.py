from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for t in nums:
            found = -1
            for x in range(t):
                if (x | (x + 1)) == t:
                    found = x
                    break
            ans.append(found)
        return ans