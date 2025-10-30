from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        s = set()
        for x in nums:
            if x - k not in s:
                s.add(x - k)
            else:
                for i in range(x - k + 1, x + k + 1):
                    if i not in s:
                        s.add(i)
                        break
        return len(s)