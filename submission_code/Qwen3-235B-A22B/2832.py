from collections import defaultdict
from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        pos_dict = defaultdict(list)
        for i, num in enumerate(nums):
            pos_dict[num].append(i)
        
        max_len = 0
        for pos_list in pos_dict.values():
            n = len(pos_list)
            left = 0
            curr_max = 0
            for right in range(n):
                # Adjust left pointer to maintain the constraint
                while (pos_list[right] - right) - (pos_list[left] - left) > k:
                    left += 1
                curr_max = max(curr_max, right - left + 1)
            max_len = max(max_len, curr_max)
        return max_len