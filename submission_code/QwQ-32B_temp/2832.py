from collections import defaultdict
from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        pos = defaultdict(list)
        for idx, num in enumerate(nums):
            pos[num].append(idx)
        
        max_len = 0
        for key in pos:
            arr = pos[key]
            n = len(arr)
            left = 0
            current_max = 0
            for right in range(n):
                # Check if the current window [left, right] is valid
                while (arr[right] - arr[left] - (right - left)) > k:
                    left += 1
                current_window = right - left + 1
                if current_window > current_max:
                    current_max = current_window
            if current_max > max_len:
                max_len = current_max
        return max_len