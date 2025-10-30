import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        
        # Step 1: Pre-process coins to remove redundant ones.
        # A coin is redundant if it's a multiple of another coin.
        coins.sort()
        filtered_coins = []
        for c in coins:
            is_redundant = False
            for fc in filtered_coins:
                if c % fc == 0:
                    is_redundant = True
                    break
            if not is_redundant:
                filtered_coins.append(c)
        
        n = len(filtered_coins)

        # Step 2: Define a function to count valid amounts up to a given value `x`.
        # This uses the Principle of Inclusion-Exclusion (PIE).
        def count_le(val: int) -> int:
            total = 0
            # Iterate over all non-empty subsets of coins using a bitmask.
            for i in range(1, 1 << n):
                current_lcm = 1
                
                # Calculate the LCM for the current subset.
                for j in range(n):
                    if (i >> j) & 1:
                        current_lcm = (current_lcm * filtered_coins[j]) // math.gcd(current_lcm, filtered_coins[j])
                        # Optimization: if LCM exceeds val, stop early for this subset.
                        if current_lcm > val:
                            break
                
                if current_lcm <= val:
                    subset_size = bin(i).count('1')
                    # Add for odd-sized subsets, subtract for even-sized ones.
                    if subset_size % 2 == 1:
                        total += val // current_lcm
                    else:
                        total -= val // current_lcm
            return total

        # Step 3: Binary search for the k-th smallest amount.
        low = 1
        high = k * filtered_coins[0]  # A safe upper bound.
        ans = high

        while low <= high:
            mid = low + (high - low) // 2
            
            if count_le(mid) >= k:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans