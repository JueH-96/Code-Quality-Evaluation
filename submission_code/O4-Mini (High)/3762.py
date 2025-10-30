from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        # Quick case: if no moves, only X=0 is feasible
        # but since m>=1 and n>=2, we handle X=0 in binary search
        # Precompute minimum point value for upper bound
        min_p = min(points)
        # Upper bound on X: in m moves, max score at any node <= m * min_p
        lo, hi = 0, m * min_p

        # Binary search for the largest X feasible
        while lo < hi:
            X = (lo + hi + 1) // 2
            # Check feasibility of achieving min score >= X
            if X == 0:
                ok = True
            elif m < n:
                # To get X>0 we need at least one visit per node => at least n moves
                ok = False
            else:
                # We'll compute the extra moves needed beyond the initial sweep of cost n
                # Compute e_last = extra visits needed at node n-1
                # r_last = ceil(X / points[n-1]), e_last = r_last - 1
                # clamp to >= 0
                p_last = points[n-1]
                r_last = (X + p_last - 1) // p_last
                e_last = r_last - 1
                if e_last < 0:
                    e_last = 0

                curr_prefix = 0   # prefix max of e[i] for i up to current k
                prev_S = 0        # previous S_k (sum of F_j up to k)
                total_rt = 0      # total round-trip cost = sum F_k * 2*(n-1-k)
                save_k = -1       # pivot index where we first have F_k>0

                # Loop k from 0 to n-2: compute M_k and F_k contributions
                # For k < n-2: M_k = max(e[0..k])
                # For k == n-2: M_{n-2} = max(e[0..n-2], e_last)
                for k in range(n-1):
                    # compute e_k = ceil(X / points[k]) - 1
                    pk = points[k]
                    r_k = (X + pk - 1) // pk
                    e_k = r_k - 1
                    if e_k < 0:
                        e_k = 0

                    if k < n-2:
                        # update prefix max up to k
                        if e_k > curr_prefix:
                            curr_prefix = e_k
                        curr_S = curr_prefix
                    else:
                        # k == n-2: include e_last in the prefix max
                        if e_k > curr_prefix:
                            curr_prefix = e_k
                        # now prefix over all nodes: max(curr_prefix, e_last)
                        if e_last > curr_prefix:
                            curr_S = e_last
                        else:
                            curr_S = curr_prefix

                    if curr_S > prev_S:
                        delta = curr_S - prev_S
                        # mark first pivot where F_k > 0
                        if save_k == -1:
                            save_k = k
                        # add round-trip cost for delta trips to pivot k
                        total_rt += delta * 2 * (n - 1 - k)
                        prev_S = curr_S

                # After the loop, prev_S == total number of round trips required
                total_rounds = prev_S  # sum F_j

                # Determine if we can save the return leg of one round-trip
                # We can only save if total_rounds > e_last (so node n-1 still gets enough visits)
                if total_rounds > e_last and save_k != -1:
                    saving = (n - 1 - save_k)
                else:
                    saving = 0

                total_moves = n + total_rt - saving
                ok = (total_moves <= m)

            if ok:
                lo = X
            else:
                hi = X - 1

        return lo