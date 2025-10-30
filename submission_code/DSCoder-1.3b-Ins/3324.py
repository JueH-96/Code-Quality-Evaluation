from typing import List

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        count = [0] * (len(nums) + 3)
        for num in nums:
            count[num] += 1
        if count[1] == 0 or count[3] == 0:
            return False
        if count[2] > count[1] and count[3] > count[2]:
            return True
        if count[2] * 2 > count[1] and count[3] > count[2] * 2:
            return True
        return False