from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = len(nums) - 1
        if len(nums) != n + 1:
            return False
        count = [0] * (n + 1)
        for num in nums:
            if num > n or count[num] > 1:
                return False
            count[num] += 1
        return count[n] == 2 and all(c == 1 for c in count[:n])