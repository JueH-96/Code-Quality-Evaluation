import math
from typing import List
from bisect import bisect_right
from collections import defaultdict

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        ratio_map = defaultdict(list)
        
        # Precompute ratio_map for all (p, q) pairs with q >= p+2
        for p in range(n):
            for q in range(p + 2, n):
                a = nums[p]
                b = nums[q]
                gcd_val = math.gcd(a, b)
                reduced = (a // gcd_val, b // gcd_val)
                ratio_map[reduced].append(q)
        
        # Sort each list in ratio_map
        for key in ratio_map:
            ratio_map[key].sort()
        
        total = 0
        
        # Iterate over all (s, r) pairs with s >= r+2
        for r in range(n):
            for s in range(r + 2, n):
                a = nums[s]
                b = nums[r]
                gcd_val = math.gcd(a, b)
                reduced = (a // gcd_val, b // gcd_val)
                if reduced not in ratio_map:
                    continue
                q_list = ratio_map[reduced]
                # Find the number of q's <= r - 2
                max_q = r - 2
                # Use bisect_right to find the insertion point of max_q + 1
                count = bisect_right(q_list, max_q)
                total += count
        
        return total