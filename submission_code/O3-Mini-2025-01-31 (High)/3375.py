from math import gcd
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # It is useful to remove coins that are “redundant”
        # (i.e. if a smaller coin divides a larger coin, then the larger coin's multiples
        # are already present). This speeds up the inclusion–exclusion computation.
        coins.sort()
        filtered = []
        for coin in coins:
            if not any(coin % prev == 0 for prev in filtered):
                filtered.append(coin)
        coins = filtered
        n = len(coins)
        
        # Precompute (lcm, sign) for every non-empty subset of coins.
        # Each coin set produces multiples of lcm(coin1,...,coin_m)
        # and its contribution in the inclusion–exclusion formula is:
        #   sign = (-1)^(m-1)   (i.e. +1 if m is odd; -1 if m is even)
        pairs = []
        
        def dfs(idx: int, curr_lcm: int, r: int) -> None:
            # r: how many coins have been chosen so far.
            # For each coin at index j starting from idx, update lcm and record that subset.
            for j in range(idx, n):
                # Compute lcm(current, coins[j])
                new_lcm = curr_lcm * coins[j] // gcd(curr_lcm, coins[j])
                sign = 1 if (r + 1) % 2 == 1 else -1  # because subset size will be r+1.
                pairs.append((new_lcm, sign))
                dfs(j + 1, new_lcm, r + 1)
        
        dfs(0, 1, 0)
        
        # Given a candidate x, count how many unique amounts (i.e. multiples) are <= x.
        # Using inclusion–exclusion:
        #   count(x) = Σ_{nonempty subset S ⊆ coins} (-1)^(|S|-1) * floor(x / lcm(S))
        def count(x: int) -> int:
            total = 0
            for l, s in pairs:
                total += s * (x // l)
            return total
        
        # Binary search for the smallest x such that count(x) >= k.
        # Lower bound: 1 (minimum positive amount), 
        # Upper bound: min(coins) * k since the kth multiple of the smallest coin is always in the union.
        lo, hi = 1, coins[0] * k
        while lo < hi:
            mid = (lo + hi) // 2
            if count(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo