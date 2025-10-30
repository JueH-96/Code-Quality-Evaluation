import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Filter out coins that are multiples of any other coin
        filtered = []
        for c in coins:
            divisible = False
            for d in coins:
                if d != c and c % d == 0:
                    divisible = True
                    break
            if not divisible:
                filtered.append(c)
        
        # Generate all non-empty subsets and their LCM and size
        n = len(filtered)
        subsets = []
        for mask in range(1, 1 << n):
            subset = []
            for i in range(n):
                if mask & (1 << i):
                    subset.append(filtered[i])
            # Compute LCM of the subset
            lcm = 1
            for num in subset:
                lcm = lcm * num // math.gcd(lcm, num)
            subsets.append((len(subset), lcm))
        
        # Binary search to find the k-th smallest value
        max_coin = max(filtered)
        low = 0
        high = k * max_coin  # Initial upper bound
        
        while low <= high:
            mid = (low + high) // 2
            count = 0
            for size, lcm in subsets:
                if lcm > mid:
                    continue
                term = mid // lcm
                if size % 2 == 1:
                    count += term
                else:
                    count -= term
            if count < k:
                low = mid + 1
            else:
                high = mid - 1
        
        return low