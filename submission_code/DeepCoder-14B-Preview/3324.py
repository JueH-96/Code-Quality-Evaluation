from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        count = Counter(nums)
        for k, v in count.items():
            if v > 2:
                return False
        k_val = len(nums) // 2
        if len(count) < k_val:
            return False
        return True