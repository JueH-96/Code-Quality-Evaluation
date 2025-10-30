import bisect
from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort the coins by their starting position
        coins.sort(key=lambda x: x[0])
        n = len(coins)
        if n == 0:
            return 0
        
        # Precompute prefix sums
        sum_c = [0] * (n + 1)
        sum_cR = [0] * (n + 1)  # sum(c_i * (R_i + 1))
        sum_cL = [0] * (n + 1)  # sum(c_i * L_i)
        for i in range(n):
            l, r, c = coins[i]
            sum_c[i+1] = sum_c[i] + c
            sum_cR[i+1] = sum_cR[i] + c * (r + 1)
            sum_cL[i+1] = sum_cL[i] + c * l
        
        # Generate candidate x values
        candidates = set()
        # Add L_i and R_i -k +1 for each segment
        for l, r, c in coins:
            candidates.add(l)
            x = r - k + 1
            candidates.add(x)
        # Add gap starts
        prev_r = None
        for l, r, c in coins:
            if prev_r is not None:
                candidates.add(prev_r + 1)
            prev_r = r
        if prev_r is not None:
            candidates.add(prev_r + 1)
        candidates.add(1)  # start of the number line
        
        # Filter x >=1 and convert to list
        candidates = [x for x in candidates if x >= 1]
        if not candidates:
            return 0
        
        # Prepare L and R arrays for binary search
        L = [seg[0] for seg in coins]
        R = [seg[1] for seg in coins]
        
        max_sum = 0
        for x in candidates:
            a = x
            b = x + k - 1
            # Find s: first index where R_i >=a
            s = bisect.bisect_left(R, a)
            # Find e: last index where L_i <=b
            e = bisect.bisect_right(L, b) - 1
            if s > e:
                continue  # no overlapping segments
            # Compute sum1
            # split_r is the first index in [s, e] where R_i >b
            split_r = bisect.bisect_left(R, b + 1, s, e + 1)
            sum1_part1 = sum_cR[split_r] - sum_cR[s]
            sum1_part2 = (b + 1) * (sum_c[e + 1] - sum_c[split_r])
            sum1 = sum1_part1 + sum1_part2
            # Compute sum2
            # split_l is the first index in [s, e] where L_i >a
            split_l = bisect.bisect_left(L, a + 1, s, e + 1)
            sum2_part1 = a * (sum_c[split_l] - sum_c[s])
            sum2_part2 = sum_cL[e + 1] - sum_cL[split_l]
            sum2 = sum2_part1 + sum2_part2
            current_sum = sum1 - sum2
            if current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum