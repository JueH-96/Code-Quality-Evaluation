from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        max_num = 0
        if not nums:
            return False
        max_num = max(nums)
        n = max_num
        base_n = list(range(1, n)) + [n] * 2 if n > 0 else [1, 1] if n == 1 else []
        if n == 0:
            base_n = []
        elif n > 1:
            base_n = list(range(1, n)) + [n] * 2
        elif n == 1:
            base_n = [1, 1]
        else:
            base_n = []

        if len(nums) != len(base_n):
            return False

        nums_counts = Counter(nums)
        base_n_counts = Counter(base_n)

        return nums_counts == base_n_counts