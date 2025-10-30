from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        freq_map = {}
        left = 0
        max_freq = 0
        result = 0
        
        for right in range(len(nums)):
            num = nums[right]
            freq_map[num] = freq_map.get(num, 0) + 1
            max_freq = max(max_freq, freq_map[num])
            
            # If deletions needed are more than k, move the left pointer
            while (right - left + 1) - max_freq > k:
                freq_map[nums[left]] -= 1
                if freq_map[nums[left]] == 0:
                    del freq_map[nums[left]]
                # If the frequency of the element being removed was equal to max_freq, recalculate max_freq
                if freq_map.get(nums[left], 0) == max_freq:
                    max_freq = max(freq_map.values()) if freq_map else 0
                left += 1
            
            # Update the result with the maximum frequency in the current window
            result = max(result, max_freq)
        
        return result