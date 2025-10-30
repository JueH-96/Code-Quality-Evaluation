class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        from collections import Counter
        frequency = Counter(nums)
        max_freq = max(frequency.values())
        return sum(freq for freq in frequency.values() if freq == max_freq)