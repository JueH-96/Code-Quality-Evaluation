from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pair_count = n // 2
        diff_freq = {}
        
        for i in range(pair_count):
            a = nums[i]
            b = nums[n - i - 1]
            diff = abs(a - b)
            if diff in diff_freq:
                diff_freq[diff] += 1
            else:
                diff_freq[diff] = 1
        
        max_freq = max(diff_freq.values()) if diff_freq else 0
        min_changes = pair_count - max_freq
        
        return min_changes