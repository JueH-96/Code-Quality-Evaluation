from math import gcd
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Helper to compute lcm of two integers
        def lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b)
        
        n = len(coins)
        
        # Pre-calculate all non-empty subsets' lcms and the parity of subset size.
        subsets = []  # each element will be (lcm_val, sign)
        
        # Use recursion to generate all subsets.
        def dfs(index: int, curr_lcm: int, count: int):
            for i in range(index, n):
                new_lcm = curr_lcm if curr_lcm == 0 else curr_lcm
                if curr_lcm == 0:
                    new_lcm = coins[i]
                else:
                    new_lcm = lcm(curr_lcm, coins[i])
                subsets.append((new_lcm, 1 if (count + 1) % 2 == 1 else -1))
                dfs(i + 1, new_lcm, count + 1)
        
        dfs(0, 0, 0)
        
        # Count how many unique values (multiples of any coin) <= x using inclusion-exclusion.
        def count(x: int) -> int:
            total = 0
            for lcm_val, sign in subsets:
                if lcm_val == 0:  # Shouldn't happen.
                    continue
                total += sign * (x // lcm_val)
            return total
        
        # Binary search for the smallest x such that count(x) >= k
        lo = 1
        hi = 10**18  # high bound
        while lo < hi:
            mid = (lo + hi) // 2
            if count(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo