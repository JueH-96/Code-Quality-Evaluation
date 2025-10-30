from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        expected_length = n + 1
        if len(nums) != expected_length:
            return False
        count = {}
        for num in nums:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
        for i in range(1, n):
            if i not in count or count[i] != 1:
                return False
        if n not in count or count[n] != 2:
            return False
        return True