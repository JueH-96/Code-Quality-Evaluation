from typing import List
from collections import defaultdict

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        freq_map = defaultdict(int)
        for num in nums:
            freq_map[num] += 1
        
        max_freq = max(freq_map.values()) if freq_map else 0
        
        total = 0
        for count in freq_map.values():
            if count == max_freq:
                total += count
                
        return total