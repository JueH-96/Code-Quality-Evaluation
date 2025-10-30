import math
from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort coins by start position
        coins.sort(key=lambda x: x[0])
        
        # Build list of constant-value intervals including gaps
        intervals = []
        pos = 1
        for seg in coins:
            l, r, c = seg
            if pos < l:
                # Add zero-value gap interval
                intervals.append((pos, l - 1, 0))
            # Add the segment interval
            intervals.append((l, r, c))
            pos = r + 1
        
        # Add a large zero-value interval at the end to handle large k
        INF = 2000000000
        intervals.append((pos, INF, 0))
        
        m = len(intervals)
        
        # Compute cumulative sum up to the end of each interval
        cum_sum = [0] * m
        start0, end0, val0 = intervals[0]
        len0 = end0 - start0 + 1
        cum_sum[0] = val0 * len0
        for i in range(1, m):
            start_i, end_i, val_i = intervals[i]
            len_i = end_i - start_i + 1
            cum_sum[i] = cum_sum[i - 1] + val_i * len_i
        
        # Define a helper function to compute prefix sum S[p]
        def get_S(p):
            if p < 1:
                return 0
            # Binary search to find the interval containing p
            left = 0
            right = m - 1
            while left <= right:
                mid = (left + right) // 2
                start_mid, end_mid, val_mid = intervals[mid]
                if start_mid <= p <= end_mid:
                    if mid == 0:
                        s_minus1_sum = 0
                    else:
                        s_minus1_sum = cum_sum[mid - 1]
                    start_pos = start_mid
                    return s_minus1_sum + val_mid * (p - start_pos + 1)
                elif p < start_mid:
                    right = mid - 1
                else:  # p > end_mid
                    left = mid + 1
            # Should not reach here if intervals cover all positions
            raise ValueError(f"Position {p} not found in intervals")
        
        # Find all change points (start positions of intervals)
        change_points = [intervals[i][0] for i in range(m)]
        
        # Generate candidate starting positions L
        candidate_L = set()
        for P in change_points:
            L1 = P
            L2 = P - k
            if L1 >= 1:
                candidate_L.add(L1)
            if L2 >= 1:
                candidate_L.add(L2)
        
        # Compute the maximum sum over all candidate L
        max_sum_val = 0
        for L in candidate_L:
            sum_win = get_S(L + k - 1) - get_S(L - 1)
            if sum_win > max_sum_val:
                max_sum_val = sum_win
        
        return max_sum_val