import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Filter out coins that are multiples of smaller coins
        coins.sort()
        filtered = []
        for coin in coins:
            is_multiple = False
            for f in filtered:
                if coin % f == 0:
                    is_multiple = True
                    break
            if not is_multiple:
                filtered.append(coin)
        coins = filtered
        n = len(coins)
        
        # Generate all non-empty subsets and their LCM and coefficients
        subsets = []
        for mask in range(1, 1 << n):
            selected = []
            for i in range(n):
                if mask & (1 << i):
                    selected.append(coins[i])
            lcm_val = 1
            for num in selected:
                gcd_val = math.gcd(lcm_val, num)
                lcm_val = (lcm_val * num) // gcd_val
            size = len(selected)
            coeff = 1 if (size % 2 == 1) else -1
            subsets.append((lcm_val, coeff))
        
        # Function to calculate the count of numbers <= x divisible by any coin
        def f(x):
            count = 0
            for (lcm_val, c) in subsets:
                count += c * (x // lcm_val)
            return count
        
        # Binary search to find the smallest x where f(x) >= k
        low = 1
        high = 1
        while f(high) < k:
            high *= 2
        
        while low < high:
            mid = (low + high) // 2
            if f(mid) < k:
                low = mid + 1
            else:
                high = mid
        
        return low