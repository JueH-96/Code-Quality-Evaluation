from typing import List

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        freq_map = {}
        for num in nums:
            freq_map[num] = freq_map.get(num, 0) + 1
        max_freq = 0
        for num in freq_map:
            max_freq = max(max_freq, freq_map[num])
        total_freq = 0
        for num in freq_map:
            if freq_map[num] == max_freq:
                total_freq += freq_map[num]
        return total_freq