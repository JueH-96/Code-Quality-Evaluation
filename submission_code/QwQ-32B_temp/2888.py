from collections import Counter
from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        counts = Counter(nums)
        dom = max(counts, key=lambda x: counts[x])
        total_dom = counts[dom]
        n = len(nums)
        current_left = 0
        
        for i in range(n):
            if nums[i] == dom:
                current_left += 1
            if i < n - 1:
                left_len = i + 1
                left_ok = (current_left * 2) > left_len
                right_count = total_dom - current_left
                right_len = n - (i + 1)
                right_ok = (right_count * 2) > right_len
                if left_ok and right_ok:
                    return i
        return -1