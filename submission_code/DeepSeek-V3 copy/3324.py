from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        count = Counter(nums)
        for num in count:
            if count[num] > 2:
                return False
        return True