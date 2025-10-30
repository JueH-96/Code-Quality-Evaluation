from typing import List
import math
from functools import lru_cache

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        # We need a fast way to get the count of a given value in any range.
        # Since nums[i] are in [1, 1000], we build prefix frequency arrays for each value.
        max_val = 1000
        # freq[val][i] will be count of occurrences of val in nums[:i]
        freq = [[0]*(n+1) for _ in range(max_val+1)]
        for i in range(n):
            val = nums[i]
            for v in range(1, max_val+1):
                freq[v][i+1] = freq[v][i]
            freq[val][i+1] += 1
        
        # Function to get divisors of a number using caching.
        @lru_cache(maxsize=None)
        def get_divisors(x: int):
            divs = []
            # iterate up to sqrt(x)
            r = int(math.isqrt(x))
            for i in range(1, r+1):
                if x % i == 0:
                    divs.append(i)
                    if i != x // i:
                        divs.append(x // i)
            return divs
        
        # We need quadruple (p, q, r, s) with p < q < r < s and gaps:
        #   q - p > 1  -> p <= q - 2,
        #   r - q > 1  -> r >= q + 2,
        #   s - r > 1  -> s >= r + 2.
        # Also note:
        #   For given q and s, valid indices:
        #       p in [0, q-2] and r in [q+2, s-2] (with q and s satisfying further that s >= q+4).
        # And the condition: nums[p] * nums[r] == nums[q] * nums[s].
        # For fixed q and s, let X = nums[q]*nums[s]. Then for any valid p and r we require
        #   nums[p] * nums[r] == X.
        # We can “split” X into two factors: d and X//d, and sum over choices:
        #   Count_p = (# of indices p in [0, q-2] with nums[p]== d)
        #   Count_r = (# of indices r in [q+2, s-2] with nums[r]== X//d)
        #
        # We iterate over all valid q and s; for each we aggregate over divisor pairs.
        
        total = 0
        # q must be at least index 2 (so that there is room for p in [0, q-2])
        # s must be at most n-1, and we require s >= q+4 (since then r in [q+2, s-2] is non-empty).
        for q in range(2, n-3):
            for s in range(q+4, n):
                X = nums[q] * nums[s]
                # Using cached divisors for X
                divs = get_divisors(X)
                for d in divs:
                    other = X // d
                    # Only consider if both factors are in the allowed range [1,1000]
                    if 1 <= d <= max_val and 1 <= other <= max_val:
                        # Count valid p indices:
                        # p in [0, q-2] means indices less than (q-1) in our prefix frequency array.
                        count_p = freq[d][q-1]  # because freq[d] is built on indices 0..(q-2) when index = q-1.
                        
                        # Count valid r indices:
                        # r in [q+2, s-2]. Our prefix freq array freq[other] counts occurrences in nums[:i].
                        # So count in [L, R] is freq[other][R+1] - freq[other][L]
                        # Here, L = q+2, R = s-2.
                        if s - 2 >= q + 2:
                            count_r = freq[other][s-1] - freq[other][q+1]
                        else:
                            count_r = 0
                        total += count_p * count_r
                    # Also consider the possibility of swapped roles: d and other swapped.
                    # But note: if d == other, then we already counted that combination.
                    if d != other and 1 <= other <= max_val and 1 <= d <= max_val:
                        # Now treat d as value for r and other as value for p.
                        count_p = freq[other][q-1]  # p with value = other
                        if s - 2 >= q + 2:
                            count_r = freq[d][s-1] - freq[d][q+1]
                        else:
                            count_r = 0
                        total += count_p * count_r
        return total