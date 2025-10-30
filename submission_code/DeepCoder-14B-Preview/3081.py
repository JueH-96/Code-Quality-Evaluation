from collections import Counter

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        if not nums:
            return 0
        freq = Counter(nums)
        max_freq = max(freq.values())
        n = len(nums)
        max_pairs = min(n // 2, n - max_freq)
        return n - 2 * max_pairs