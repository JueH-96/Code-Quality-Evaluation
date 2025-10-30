from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        res = []
        for num in nums:
            if num == 2:
                res.append(-1)
                continue
            found = False
            for x in range(num):
                if (x | (x + 1)) == num:
                    res.append(x)
                    found = True
                    break
            if not found:
                res.append(-1)
        return res