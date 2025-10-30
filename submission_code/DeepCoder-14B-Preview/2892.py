from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        if not nums:
            return False  # Though constraints say nums is non-empty
        max_val = max(nums)
        n = max_val
        if len(nums) != n + 1:
            return False
        count = Counter(nums)
        for i in range(1, n):
            if count.get(i, 0) != 1:
                return False
        if count.get(n, 0) != 2:
            return False
        return True