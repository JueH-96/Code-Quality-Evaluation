from typing import List
from math import gcd

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        n = len(coins)
        # Precompute all non-empty subsets with their LCM and subset size (number of coins)
        subsets = []
        for mask in range(1, 1 << n):
            current_lcm = 1
            bits = 0
            for i in range(n):
                if mask & (1 << i):
                    current_lcm = lcm(current_lcm, coins[i])
                    bits += 1
            subsets.append((current_lcm, bits))
        
        # Binary search boundaries
        low = 1
        high = max(coins) * k
        
        def count(x):
            res = 0
            for current_lcm, bits in subsets:
                if current_lcm <= x:
                    if bits % 2 == 1:
                        res += x // current_lcm
                    else:
                        res -= x // current_lcm
            return res
        
        # Perform binary search
        while low < high:
            mid = (low + high) // 2
            c = count(mid)
            if c < k:
                low = mid + 1
            else:
                high = mid
        
        return low