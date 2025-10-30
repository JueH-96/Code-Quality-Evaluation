class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        from collections import Counter
        freq = Counter(nums)
        maxFreq = max(freq.values())
        return sum(count for count in freq.values() if count == maxFreq)