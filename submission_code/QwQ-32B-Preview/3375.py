import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Define the count function using inclusion-exclusion
        def count(x):
            total = 0
            # Iterate over all non-empty subsets of coins using bit manipulation
            for mask in range(1, 1 << len(coins)):
                subset = [coins[i] for i in range(len(coins)) if mask & (1 << i)]
                lcm_val = math.lcm(*subset)
                subset_size = bin(mask).count('1')
                # Add or subtract based on the size of the subset
                if subset_size % 2 == 1:
                    total += x // lcm_val
                else:
                    total -= x // lcm_val
            return total
        
        # Binary search setup
        low = min(coins)
        high = k * min(coins)
        
        # Refine the high bound to be more efficient
        while count(high) < k:
            low = high
            high *= 2
        
        # Binary search to find the smallest x where count(x) >= k
        while low < high:
            mid = (low + high) // 2
            if count(mid) >= k:
                high = mid
            else:
                low = mid + 1
        
        return low