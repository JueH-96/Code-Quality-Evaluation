class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        from collections import Counter
        
        freq = Counter(nums)
        max_freq = max(freq.values())
        
        # Sum the frequency of all elements that have the maximum frequency
        return sum(count for count in freq.values() if count == max_freq)