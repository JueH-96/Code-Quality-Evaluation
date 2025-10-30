from bisect import bisect_left
from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        # Enrich intervals with original index.
        enriched = [(l, r, w, i) for i, (l, r, w) in enumerate(intervals)]
        # Sort intervals by r, then by l, then by original index (for deterministic behavior)
        enriched.sort(key=lambda x: (x[1], x[0], x[3]))
        
        # Precompute list of ending times (r) for binary search.
        ends = [interval[1] for interval in enriched]
        
        # For each interval i (in sorted order), find the index p (largest index with r < l_i)
        # We use binary search: find leftmost index where end >= l, then subtract one.
        p_indices = []
        for i, (l, r, w, orig) in enumerate(enriched):
            # Need r < l, so use bisect_left on ends with l.
            j = bisect_left(ends, l)
            p = j - 1
            p_indices.append(p)
        
        # We want to choose up to 4 non-overlapping intervals.
        K = 4

        # dp[i][k] = best tuple (total_weight, lex smallest selected original indices list)
        # considering first i intervals (i from 0 to n) and selecting exactly k intervals.
        # For convenience, we build dp table with dimensions (n+1) x (K+1).
        # We use "impossible" state as None.
        dp = [[None] * (K + 1) for _ in range(n + 1)]
        # Base: dp[0][0] is (0, []) and dp[0][k] for k>=1 is impossible.
        dp[0][0] = (0, [])
        for k in range(1, K + 1):
            dp[0][k] = (-1, [])  # -1 indicates impossible (or you can use -inf)
        
        # Fill dp table:
        for i in range(1, n + 1):
            # Current interval index in sorted order is i-1.
            l, r, weight, orig = enriched[i - 1]
            p = p_indices[i - 1]
            for k in range(K + 1):
                # Option 1: not take i-th interval => dp[i-1][k]
                best = dp[i - 1][k]
                # Option 2: take i-th interval if k>=1 and if preceding state is valid.
                if k > 0:
                    # For previous state, use index p+1 because dp index is 1-indexed (dp[?])
                    prev_state = dp[p + 1][k - 1] if (p + 1) >= 0 else None
                    if prev_state is not None and prev_state[0] != -1:
                        candidate_sum = prev_state[0] + weight
                        # Build new candidate's selected original indices list.
                        candidate_list = prev_state[1] + [orig]
                        # Our lex ordering: we compare candidate_list as standard list.
                        candidate = (candidate_sum, candidate_list)
                        # Now, choose the best between best and candidate.
                        if best is None or candidate_sum > best[0]:
                            best = candidate
                        elif candidate_sum == best[0]:
                            # Compare lex: because we want lexicographically smallest list.
                            if candidate_list < best[1]:
                                best = candidate
                # Also, if best is still None, set it as impossible
                if best is None:
                    best = (-1, [])
                dp[i][k] = best
        
        # Now, among dp[n][0..K], choose the one with maximum sum.
        answer = (-1, [])
        for k in range(K + 1):
            candidate = dp[n][k]
            if candidate[0] > answer[0]:
                answer = candidate
            elif candidate[0] == answer[0] and candidate[1] < answer[1]:
                answer = candidate
        
        # The resulting answer list might not be sorted, 
        # but note: intervals chosen are arbitrary order - however, 
        # by construction our dp maintains order of selection based on increasing r 
        # which is not equal to increasing original index.
        # The problem asks for lexicographically smallest array of chosen indices.
        # So we sort the indices (the array length is at most 4, so that's fine).
        ans_list = sorted(answer[1])
        return ans_list


# Simple testing (you can remove or comment out these before submission)
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    intervals1 = [[1,3,2],[4,5,2],[1,5,5],[6,9,3],[6,7,1],[8,9,1]]
    print(sol.maximumWeight(intervals1))  # Expected: [2,3]

    # Example 2
    intervals2 = [[5,8,1],[6,7,7],[4,7,3],[9,10,6],[7,8,2],[11,14,3],[3,5,5]]
    print(sol.maximumWeight(intervals2))  # Expected: [1,3,5,6]