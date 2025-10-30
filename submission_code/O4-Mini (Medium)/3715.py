from bisect import bisect_left, bisect_right
from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort segments by left endpoint
        coins.sort(key=lambda x: x[0])
        n = len(coins)
        L = [seg[0] for seg in coins]
        R = [seg[1] for seg in coins]
        C = [seg[2] for seg in coins]
        # prefix sum of full segment contributions = c_i * length_i
        pref = [0] * n
        for i in range(n):
            length = R[i] - L[i] + 1
            val = length * C[i]
            pref[i] = val + (pref[i-1] if i > 0 else 0)
        
        # Generate candidate window start positions x
        # Only at points where the overlap function can change slope:
        # x = l_i - k + 1, l_i, r_i - k + 1, r_i + 1
        cand = set()
        for i in range(n):
            l, r = L[i], R[i]
            cand.add(l)
            cand.add(l - k + 1)
            cand.add(r - k + 1)
            cand.add(r + 1)
        # Convert to sorted list
        xs = sorted(cand)
        
        def window_sum(x: int) -> int:
            # window = [x, x+k-1]
            y = x + k - 1
            # find last segment with L[i] <= y
            i = bisect_right(L, y) - 1
            if i < 0:
                return 0
            # find first segment with R[j] >= x
            j = bisect_left(R, x)
            if j > i:
                return 0
            # if only one segment overlaps
            if i == j:
                ov = min(R[i], y) - max(L[i], x) + 1
                return max(0, ov) * C[i]
            # partial on j
            ovj = min(R[j], y) - max(L[j], x) + 1
            sum_j = max(0, ovj) * C[j]
            # partial on i
            ovi = min(R[i], y) - max(L[i], x) + 1
            sum_i = max(0, ovi) * C[i]
            # full segments between j+1 and i-1
            if j + 1 <= i - 1:
                sum_full = pref[i-1] - pref[j]
            else:
                sum_full = 0
            return sum_j + sum_full + sum_i
        
        ans = 0
        for x in xs:
            # No need to consider windows that lie entirely outside all segments
            # but our window_sum returns 0 in those cases.
            s = window_sum(x)
            if s > ans:
                ans = s
        return ans