import bisect
from collections import Counter

class Solution:
    def maxFrequency(self, nums: list, k: int, numOperations: int) -> int:
        if not nums:
            return 0
        nums_sorted = sorted(nums)
        freq = Counter(nums)
        min_val = min(nums)
        max_val = max(nums)
        L0 = min_val - k
        R0 = max_val + k
        ans = 0
        
        for x in range(L0, R0 + 1):
            left_bound = x - k
            right_bound = x + k
            l_idx = bisect.bisect_left(nums_sorted, left_bound)
            r_idx = bisect.bisect_right(nums_sorted, right_bound)
            count = r_idx - l_idx
            orig_freq = freq.get(x, 0)
            current = min(orig_freq + numOperations, count)
            if current > ans:
                ans = current
        return ans