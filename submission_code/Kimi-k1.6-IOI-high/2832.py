from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        positions = defaultdict(list)
        for idx, num in enumerate(nums):
            positions[num].append(idx)
        
        max_len = 0
        for num in positions:
            arr = positions[num]
            i = 0
            for j in range(len(arr)):
                # Calculate the required deletions for the window [i, j]
                while (arr[j] - arr[i] - (j - i)) > k:
                    i += 1
                current_len = j - i + 1
                if current_len > max_len:
                    max_len = current_len
        return max_len