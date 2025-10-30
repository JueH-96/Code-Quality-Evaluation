from bisect import bisect_left
from collections import defaultdict
from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n // 2
        pairs = []
        for i in range(m):
            a = nums[i]
            b = nums[n - 1 - i]
            pairs.append((a, b))
        
        diff_count = defaultdict(int)
        max_vals = []
        
        for a, b in pairs:
            original_diff = abs(a - b)
            M1 = max(b, k - b)
            M2 = max(a, k - a)
            max_val = max(M1, M2)
            diff_count[original_diff] += 1
            max_vals.append(max_val)
        
        max_vals.sort()
        max_current = -1
        
        for X in range(k + 1):
            C0 = diff_count.get(X, 0)
            idx = bisect_left(max_vals, X)
            S = len(max_vals) - idx
            current = C0 + S
            if current > max_current:
                max_current = current
        
        return 2 * m - max_current