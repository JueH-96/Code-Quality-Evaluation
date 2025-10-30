from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        MAX = 100000
        freq = [0] * (MAX + 1)
        for num in nums:
            if 1 <= num <= MAX:
                freq[num] += 1
        
        prefix = [0] * (MAX + 2)
        for i in range(1, MAX + 1):
            prefix[i] = prefix[i - 1] + freq[i]
        
        max_freq = 0
        min_t = 1
        max_t = MAX
        
        for t in range(min_t, max_t + 1):
            left = max(t - k, 1)
            right = min(t + k, MAX)
            num_in_range = prefix[right] - prefix[left - 1]
            num_equal = freq[t]
            if num_in_range - num_equal <= numOperations:
                current_freq = num_in_range
            else:
                current_freq = num_equal + numOperations
            if current_freq > max_freq:
                max_freq = current_freq
                
        return max_freq