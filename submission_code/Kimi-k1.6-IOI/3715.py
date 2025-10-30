import bisect
from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort the coins by their starting position
        coins.sort()
        n = len(coins)
        if n == 0:
            return 0
        
        l_list = [coin[0] for coin in coins]
        r_list = [coin[1] for coin in coins]
        c_list = [coin[2] for coin in coins]
        
        # Generate all candidate starting positions
        candidates = set()
        for l, r, c in coins:
            candidates.add(l)
            candidates.add(r + 1)
            candidates.add(max(1, l - k + 1))
            candidates.add(max(1, r - k + 1))
        
        # Convert candidates to a sorted list
        candidates = sorted(candidates)
        
        # Precompute prefix sums for c, c*l, c*r
        prefix_c = [0] * (n + 1)
        prefix_lr = [0] * (n + 1)
        prefix_rr = [0] * (n + 1)
        for i in range(n):
            prefix_c[i+1] = prefix_c[i] + c_list[i]
            prefix_lr[i+1] = prefix_lr[i] + c_list[i] * l_list[i]
            prefix_rr[i+1] = prefix_rr[i] + c_list[i] * r_list[i]
        
        max_sum = 0
        for s in candidates:
            L = s
            R = s + k - 1
            # Find the end index in l_list where l_i <= R
            end_idx = bisect.bisect_right(l_list, R) - 1
            if end_idx < 0:
                current_sum = 0
            else:
                # Find the start index in r_list where r_i >= L within 0 to end_idx
                start_idx = bisect.bisect_left(r_list, L, 0, end_idx + 1)
                if start_idx > end_idx:
                    current_sum = 0
                else:
                    # Calculate sum3 (sum of c_i)
                    sum3 = prefix_c[end_idx + 1] - prefix_c[start_idx]
                    # Calculate sum1 (sum of c_i * min(R, r_i))
                    # Find the first index a where r_i > R in [start_idx, end_idx]
                    a = bisect.bisect_right(r_list, R, start_idx, end_idx + 1)
                    sum1_part = prefix_rr[a] - prefix_rr[start_idx]
                    sum1 = sum1_part + R * (prefix_c[end_idx + 1] - prefix_c[a])
                    # Calculate sum2 (sum of c_i * max(L, l_i))
                    # Find the first index b where l_i >= L in [start_idx, end_idx]
                    b = bisect.bisect_left(l_list, L, start_idx, end_idx + 1)
                    sum2_part = prefix_lr[end_idx + 1] - prefix_lr[b]
                    sum2 = L * (prefix_c[b] - prefix_c[start_idx]) + sum2_part
                    current_sum = sum1 - sum2 + sum3
            if current_sum > max_sum:
                max_sum = current_sum
        return max_sum