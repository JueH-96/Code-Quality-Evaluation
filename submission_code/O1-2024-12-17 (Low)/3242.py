class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        from collections import Counter
        # Count the frequency of each element
        freq = Counter(nums)
        # Determine the maximum frequency
        max_freq = max(freq.values())
        # Sum the frequencies of all elements that have the maximum frequency
        return sum(count for count in freq.values() if count == max_freq)