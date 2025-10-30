from typing import List

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # Find the maximum frequency
        max_freq = max(freq.values())
        
        # Sum up the occurrences of all elements that have the maximum frequency
        total = 0
        for num, count in freq.items():
            if count == max_freq:
                total += count
        
        return total