from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for x in nums:
            found = False
            # We only need to check up to x-1 because x OR (x+1) > x
            for A in range(x):
                if (A | (A + 1)) == x:
                    ans.append(A)
                    found = True
                    break
            if not found:
                ans.append(-1)
        return ans