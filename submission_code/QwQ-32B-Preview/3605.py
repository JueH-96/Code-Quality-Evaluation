from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for prime in nums:
            found = False
            for x in range(prime + 1):
                if x | (x + 1) == prime:
                    ans.append(x)
                    found = True
                    break
            if not found:
                ans.append(-1)
        return ans