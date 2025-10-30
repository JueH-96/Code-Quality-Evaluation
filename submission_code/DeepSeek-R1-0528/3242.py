from collections import defaultdict

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        freq_map = defaultdict(int)
        for num in nums:
            freq_map[num] += 1
        
        max_freq = 0
        for count in freq_map.values():
            if count > max_freq:
                max_freq = count
        
        count_max = 0
        for count in freq_map.values():
            if count == max_freq:
                count_max += 1
        
        return count_max * max_freq