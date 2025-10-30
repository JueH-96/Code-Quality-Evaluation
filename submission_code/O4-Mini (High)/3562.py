from typing import List
from bisect import bisect_left

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Prepare intervals sorted by end time, keep original indices
        n = len(intervals)
        arr = [(l, r, w, i) for i, (l, r, w) in enumerate(intervals)]
        arr.sort(key=lambda x: x[1])
        # 1-indexed for convenience
        arr = [None] + arr
        
        # Build list of end-times and p array: p[i] = largest j < i with arr[j].r < arr[i].l
        ends = [arr[i][1] for i in range(1, n+1)]
        p = [0] * (n+1)
        for i in range(1, n+1):
            l_i = arr[i][0]
            # bisect_left returns first idx in ends where ends[idx] >= l_i
            pos = bisect_left(ends, l_i)
            # that means ends[0..pos-1] < l_i, so p[i] = pos
            p[i] = pos
        
        # DP arrays
        NEG_INF = -10**18
        # weight_dp[i][t] = max weight using exactly t intervals among first i
        weight_dp = [[NEG_INF]*5 for _ in range(n+1)]
        # path_dp[i][t] = lexicographically smallest sorted list of original indices achieving that weight
        path_dp = [[None]*5 for _ in range(n+1)]
        
        # base case: 0 intervals => weight 0, empty path
        weight_dp[0][0] = 0
        path_dp[0][0] = []
        
        # Fill DP
        for i in range(1, n+1):
            # Exclusion: carry over dp[i-1][*]
            for t in range(5):
                weight_dp[i][t] = weight_dp[i-1][t]
                path_dp[i][t] = path_dp[i-1][t]
            
            l_i, r_i, w_i, orig_i = arr[i]
            # Try to include this interval as the t-th pick
            for t in range(1, 5):
                prev_i = p[i]
                prev_w = weight_dp[prev_i][t-1]
                if prev_w <= NEG_INF:
                    continue
                cand_w = prev_w + w_i
                curr_w = weight_dp[i][t]
                # build candidate path by inserting orig_i into the sorted previous path
                prev_path = path_dp[prev_i][t-1]
                idx_ins = bisect_left(prev_path, orig_i)
                cand_path = prev_path[:idx_ins] + [orig_i] + prev_path[idx_ins:]
                
                if cand_w > curr_w:
                    weight_dp[i][t] = cand_w
                    path_dp[i][t] = cand_path
                elif cand_w == curr_w:
                    # tie: pick lexicographically smaller
                    curr_path = path_dp[i][t]
                    if curr_path is None or cand_path < curr_path:
                        path_dp[i][t] = cand_path
        
        # Pick best among up to 4 intervals (t = 0..4)
        best_w = NEG_INF
        best_path: List[int] = []
        for t in range(5):
            w = weight_dp[n][t]
            if w > best_w:
                best_w = w
                best_path = path_dp[n][t]
            elif w == best_w:
                path = path_dp[n][t]
                if path is not None and path < best_path:
                    best_path = path
        
        return best_path