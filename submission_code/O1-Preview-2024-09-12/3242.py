class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        from collections import Counter
        c = Counter(nums)
        max_freq = max(c.values())
        total = sum(freq for freq in c.values() if freq == max_freq)
        return total