class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        count = Counter(nums)
        for v in count.values():
            if v > 2:
                return False
        return True