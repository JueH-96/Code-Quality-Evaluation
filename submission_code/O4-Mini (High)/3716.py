class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        # We keep dp_val[v][d] = best length of a subsequence ending with value v
        # whose last absolute-difference is exactly d.
        # We also keep best_suff[v][d] = max(dp_val[v][d'], for d' >= d).
        # Initialize for each value v: dp_val[v][MAX_DIFF] = 1 (single element),
        # and best_suff[v][*] = 1. Then process elements in order, for each new
        # element v we extend all seen values pv -> v with w = |v-pv|, requiring
        # last diff >= w, i.e. use best_suff[pv][w]. Finally update dp_val[v][w]
        # and recompute best_suff[v].
        MAX_VAL = 300
        MAX_DIFF = 300
        # dp_val and best_suff dimensions: [0..MAX_VAL] x [0..MAX_DIFF]
        dp_val = [[0] * (MAX_DIFF + 1) for _ in range(MAX_VAL + 1)]
        best_suff = [[1] * (MAX_DIFF + 1) for _ in range(MAX_VAL + 1)]
        # Single-element subsequence has length 1 and "last diff" = MAX_DIFF
        for v in range(MAX_VAL + 1):
            dp_val[v][MAX_DIFF] = 1

        seen = [False] * (MAX_VAL + 1)
        ans = 1

        for v in nums:
            updated = False
            # Try to extend every previously seen value pv -> v
            for pv in range(1, MAX_VAL + 1):
                if not seen[pv]:
                    continue
                w = abs(v - pv)
                new_len = best_suff[pv][w] + 1
                if new_len > dp_val[v][w]:
                    dp_val[v][w] = new_len
                    updated = True
                    if new_len > ans:
                        ans = new_len
            # If we updated dp_val[v], recompute its suffix-max array
            if updated:
                # best_suff[v][d] = max(dp_val[v][d], best_suff[v][d+1])
                for d in range(MAX_DIFF, -1, -1):
                    if d == MAX_DIFF:
                        best_suff[v][d] = dp_val[v][d]
                    else:
                        nxt = best_suff[v][d + 1]
                        best_suff[v][d] = dp_val[v][d] if dp_val[v][d] > nxt else nxt
            # Mark this value as now seen for future extensions
            seen[v] = True

        return ans