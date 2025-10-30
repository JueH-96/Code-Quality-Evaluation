from bisect import bisect_left
from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        # Augment intervals with original index
        intervals_s = []
        for i, (l, r, w) in enumerate(intervals):
            intervals_s.append((l, r, w, i))
        # Sort by end time r
        intervals_s.sort(key=lambda x: x[1])
        # Extract the sorted end times
        ends = [iv[1] for iv in intervals_s]
        # Compute p array: for each interval i, the rightmost interval j with ends[j] < l_i
        p = [ -1 ] * n
        for i in range(n):
            l_i = intervals_s[i][0]
            # find first end >= l_i
            idx = bisect_left(ends, l_i)
            p[i] = idx - 1
        
        # dp_w[j][t]: max weight using first j intervals (in sorted order), taking t intervals
        # dp_seq[j][t]: tuple of chosen original indices (sorted) for that optimal
        # j=0..n, t=0..4
        NEG_INF = -10**30
        dp_w = [[NEG_INF]*5 for _ in range(n+1)]
        dp_seq = [[() for _ in range(5)] for _ in range(n+1)]
        # base case: j=0, t=0 -> weight 0, empty seq
        dp_w[0][0] = 0
        dp_seq[0][0] = ()
        
        # Fill DP
        for j in range(1, n+1):
            # First copy over the "skip" case from j-1
            for t in range(5):
                dp_w[j][t] = dp_w[j-1][t]
                dp_seq[j][t] = dp_seq[j-1][t]
            # Now consider taking interval j-1
            l_j, r_j, w_j, orig_j = intervals_s[j-1]
            pj = p[j-1]  # this is index in 0..n-1 of the prev non-overlapping
            base_idx = pj + 1  # dp row corresponding to up to pj
            for t in range(1, 5):
                prev_w = dp_w[base_idx][t-1]
                if prev_w == NEG_INF:
                    continue
                cand_w = prev_w + w_j
                # build candidate seq: take seq from dp_seq[base_idx][t-1] and add orig_j
                prev_seq = dp_seq[base_idx][t-1]
                # insert orig_j into a sorted tuple
                # since length <=3, just do simple insert
                cand_list = list(prev_seq)
                # find position to insert to keep sorted
                import bisect
                pos = bisect.bisect_left(cand_list, orig_j)
                cand_list.insert(pos, orig_j)
                cand_seq = tuple(cand_list)
                # compare to current dp[j][t]
                if cand_w > dp_w[j][t]:
                    dp_w[j][t] = cand_w
                    dp_seq[j][t] = cand_seq
                elif cand_w == dp_w[j][t]:
                    # tie: pick lexicographically smallest
                    if cand_seq < dp_seq[j][t]:
                        dp_seq[j][t] = cand_seq
        
        # At the end, pick best over t=0..4
        best_w = NEG_INF
        best_seq = ()
        for t in range(5):
            w_t = dp_w[n][t]
            if w_t > best_w:
                best_w = w_t
                best_seq = dp_seq[n][t]
            elif w_t == best_w:
                # tie: lex compare
                if dp_seq[n][t] < best_seq:
                    best_seq = dp_seq[n][t]
        # best_seq is a tuple of original indices, already sorted
        return list(best_seq)