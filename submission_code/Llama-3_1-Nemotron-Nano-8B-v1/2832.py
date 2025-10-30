from collections import defaultdict
from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        counts = defaultdict(int)
        reverse = defaultdict(int)  # frequency of frequencies
        current_max_freq = 0
        left = 0
        ans = 0
        for right in range(len(nums)):
            num = nums[right]
            old_freq = counts[num]
            if old_freq > 0:
                reverse[old_freq] -= 1
                if reverse[old_freq] == 0:
                    del reverse[old_freq]
            counts[num] += 1
            new_freq = old_freq + 1
            reverse[new_freq] += 1
            if new_freq > current_max_freq:
                current_max_freq = new_freq
            # Check if the current window is valid
            while (right - left + 1 - current_max_freq) > k:
                left_num = nums[left]
                old_freq_left = counts[left_num]
                reverse[old_freq_left] -= 1
                if reverse[old_freq_left] == 0:
                    del reverse[old_freq_left]
                counts[left_num] -= 1
                new_freq_left = old_freq_left - 1
                if new_freq_left > 0:
                    reverse[new_freq_left] += 1
                # Update current_max_freq if needed
                if old_freq_left == current_max_freq:
                    if reverse.get(old_freq_left, 0) == 0:
                        while current_max_freq > 0 and reverse.get(current_max_freq, 0) == 0:
                            current_max_freq -= 1
                left += 1
            # Update the answer
            ans = max(ans, current_max_freq)
        return ans