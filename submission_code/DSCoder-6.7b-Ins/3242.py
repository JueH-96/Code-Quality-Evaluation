from collections import Counter

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element in the list
        freq = Counter(nums)
        
        # Find the maximum frequency
        max_freq = max(freq.values())
        
        # Count the number of elements with the maximum frequency
        max_freq_elements = sum(1 for freq in freq.values() if freq == max_freq)
        
        return max_freq_elements