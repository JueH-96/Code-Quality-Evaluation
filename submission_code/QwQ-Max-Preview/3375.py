import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        n = len(coins)
        subsets_info = []
        
        # Precompute all non-empty subsets, their LCM, and sign
        for mask in range(1, 1 << n):
            subset = []
            current_lcm = 1
            for i in range(n):
                if mask & (1 << i):
                    subset.append(coins[i])
            # Compute LCM of the subset
            for num in subset:
                current_lcm = current_lcm * num // math.gcd(current_lcm, num)
            s = len(subset)
            sign = 1 if s % 2 == 1 else -1
            subsets_info.append((current_lcm, sign))
        
        # Binary search setup
        low = 1
        high = max(coins) * k
        
        while low < high:
            mid = (low + high) // 2
            count = 0
            for lcm_val, sign in subsets_info:
                if lcm_val > mid:
                    continue
                count += (mid // lcm_val) * sign
            if count < k:
                low = mid + 1
            else:
                high = mid
        return low