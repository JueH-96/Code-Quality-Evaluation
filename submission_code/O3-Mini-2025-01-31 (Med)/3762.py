from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        # It is impossible to visit all indices if m < n (the minimal route cost).
        # In that case, answer is 0.
        if m < n:
            return 0
        
        # Check function: given candidate minimum score x, determine if
        # it is possible with at most m moves.
        def can_achieve(x: int) -> bool:
            # For each index i, compute deficit: extra visits needed beyond the
            # base one visit. We need at least k visits, with:
            #    k = ceil(x / points[i])
            # so deficit d[i] = max(0, k - 1)
            d = [0] * n
            for i in range(n):
                # Instead of math.ceil, use integer arithmetic:
                k = (x + points[i] - 1) // points[i]
                # We already get one visit from the base route.
                d[i] = 0 if k <= 1 else k - 1

            # We need to cover these deficits by “bounces” on edges between consecutive indices.
            # We have n-1 edges (between 0 and 1, 1 and 2, …, n-2 and n-1).
            L = n - 1

            best_extra = float('inf')
            # Try all possibilities for endpoint half-bounce flags.
            for h0 in (0, 1):
                for hend in (0, 1):
                    # We will assign bounce cycles t[0...L-1].
                    total = 0
                    # For index 0: the bounce from edge (0,1) (if any) plus a possible half bounce (h0)
                    t0 = max(0, d[0] - h0)
                    total += t0
                    prev = t0  # This is t[0]
                    # Now, for each interior index i = 1..n-2, we have constraint:
                    #   t[i-1] + t[i] >= d[i]
                    # so we set:
                    #   t[i] = max(0, d[i] - t[i-1])
                    t_list = [t0]
                    valid = True
                    for i in range(1, n - 1):
                        ti = max(0, d[i] - prev)
                        t_list.append(ti)
                        total += ti
                        prev = ti
                    # Now, for index n-1: constraint is t[n-2] + hend >= d[n-1].
                    if t_list[-1] < d[n-1] - hend:
                        # We can “adjust” the last bounce cycle upwards.
                        total += (d[n-1] - hend - t_list[-1])
                        t_list[-1] = d[n-1] - hend
                    # Compute extra moves cost:
                    extra_cost = 2 * total + (h0 + hend)
                    best_extra = min(best_extra, extra_cost)
            # Total moves = base route cost + extra moves.
            total_moves = n + best_extra
            return total_moves <= m

        # Binary search for the maximum x (minimum score achievable) 
        # Lower bound is 0. Upper bound can be m * min(points)
        lo = 0
        hi = m * min(points)
        answer = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_achieve(mid):
                answer = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return answer