from typing import List
from math import gcd

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Sort the coins so we can prune larger LCMs early in our DFS
        coins.sort()
        n = len(coins)
        
        # Precompute LCM via gcd to avoid overflow or large intermediate multiplication
        def lcm(a: int, b: int) -> int:
            return a // gcd(a, b) * b
        
        # Count how many distinct multiples (from any single coin) are <= x
        def count_le(x: int) -> int:
            # We'll do a DFS over subsets of coins to apply
            # inclusion-exclusion for "divisible by at least one coin"
            def dfs(start_index: int, current_lcm: int, parity: int) -> int:
                total = 0
                for i in range(start_index, n):
                    new_lcm = lcm(current_lcm, coins[i])
                    # If new_lcm > x, further coins won't help (they are >= coins[i]) -> prune
                    if new_lcm > x:
                        break
                    # If picking this coin flips us to odd subset size (parity=1), we add
                    # If even subset size (parity=0), we subtract
                    new_parity = 1 - parity
                    if new_parity == 1:
                        total += x // new_lcm
                    else:
                        total -= x // new_lcm
                    total += dfs(i + 1, new_lcm, new_parity)
                return total
            
            return dfs(0, 1, 0)
        
        # Binary search for the smallest amount "low" such that count_le(low) >= k
        low, high = 1, min(coins) * k  # k-th multiple can't exceed k * min_coin
        while low < high:
            mid = (low + high) // 2
            if count_le(mid) >= k:
                high = mid
            else:
                low = mid + 1
        
        return low