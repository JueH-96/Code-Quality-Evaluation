class Solution:
    def maxFrequencyElements(self, nums):
        from collections import Counter
        count = Counter(nums)
        max_freq = max(count.values())
        return sum(1 for val in count.values() if val == max_freq)