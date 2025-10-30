from typing import List
import bisect

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort the intervals based on their start positions
        coins.sort()
        n = len(coins)
        starts = [c[0] for c in coins]
        ends = [c[1] for c in coins]
        # Precompute prefix sums of coins
        precoin = [0] * (n + 1)
        for i in range(n):
            s, e, c = coins[i]
            precoin[i+1] = precoin[i] + (e - s + 1) * c
        
        # Collect all candidate starting positions 'a'
        candidates = set()
        for i in range(n):
            s, e, c = coins[i]
            candidates.add(s - k + 1)
            candidates.add(s)
            candidates.add(e - k + 1)
            candidates.add(e)
            candidates.add(s - 1)
            candidates.add(s + 1)
            candidates.add(e - k)
            candidates.add(e - k + 2)
            candidates.add(e + 1 - k + 1)
            candidates.add(e + 1)
            candidates.add(e - 1)
        
        max_sum = 0
        
        # Check contained sums for each interval
        for i in range(n):
            s, e, c = coins[i]
            length = e - s + 1
            current = min(k, length) * c
            if current > max_sum:
                max_sum = current
        
        # Helper function to compute sum_up_to(x)
        def compute_sum_up_to(x):
            if x < starts[0]:
                return 0
            idx = bisect.bisect_right(starts, x) - 1
            if idx < 0:
                return 0
            if ends[idx] >= x:
                return precoin[idx] + (x - starts[idx] + 1) * coins[idx][2]
            else:
                return precoin[idx + 1]
        
        # Evaluate all candidate 'a's
        for a in candidates:
            if a < 1:
                continue
            b = a + k - 1
            if b > 10**9:
                continue
            # Compute sum_up_to(b)
            sum_b = compute_sum_up_to(b)
            # Compute sum_up_to(a-1)
            sum_a_1 = compute_sum_up_to(a - 1)
            current = sum_b - sum_a_1
            if current > max_sum:
                max_sum = current
        
        return max_sum