from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        expected_length = n + 1
        if len(nums) != expected_length:
            return False
        expected_counts = Counter(range(1, n)) + Counter({n: 2})
        actual_counts = Counter(nums)
        return expected_counts == actual_counts