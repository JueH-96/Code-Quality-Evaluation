from typing import List

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        freq = {}
        max_freq = 0
        for num in nums:
            if num in freq:
                freq[num] += 1
            else:
                freq[num] = 1
            max_freq = max(max_freq, freq[num])
        
        total_freq = 0
        for count in freq.values():
            if count == max_freq:
                total_freq += count
        
        return total_freq