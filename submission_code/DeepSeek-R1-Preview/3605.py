from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for num in nums:
            min_x = -1
            for x in range(num):
                if (x | (x + 1)) == num:
                    if min_x == -1 or x < min_x:
                        min_x = x
            ans.append(min_x if min_x != -1 else -1)
        return ans