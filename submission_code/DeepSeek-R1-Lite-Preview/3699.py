from typing import List
import math
import bisect

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        ratio_q = {}
        
        # Step 1: Collect (p, q) pairs and store q's for each normalized ratio
        for p in range(n - 2):
            for q in range(p + 2, n):
                GCD = math.gcd(nums[p], nums[q])
                a = nums[p] // GCD
                b = nums[q] // GCD
                ratio = (a, b)
                if ratio not in ratio_q:
                    ratio_q[ratio] = []
                ratio_q[ratio].append(q)
        
        total = 0
        
        # Step 2: Iterate through (r, s) pairs and count valid (p, q) pairs
        for r in range(2, n):
            for s in range(r + 2, n):
                GCD = math.gcd(nums[s], nums[r])
                a = nums[s] // GCD
                b = nums[r] // GCD
                ratio = (a, b)
                if ratio in ratio_q:
                    # Count number of q's < r for this ratio
                    q_list = ratio_q[ratio]
                    count = bisect.bisect_left(q_list, r)
                    total += count
        
        return total