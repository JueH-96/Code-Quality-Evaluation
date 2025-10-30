import bisect

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        intervals_with_idx = []
        for i, interval in enumerate(intervals):
            l, r, w = interval
            intervals_with_idx.append((l, r, w, i))
        
        intervals_sorted = sorted(intervals_with_idx, key=lambda x: (x[1], x[0]))
        n = len(intervals_sorted)
        R = [x[1] for x in intervals_sorted]
        
        INF_NEG = -10**18
        dp = [[(INF_NEG, None)] * n for _ in range(5)]
        
        for j in range(n):
            dp[0][j] = (0, ())
        
        for j in range(n):
            l_j, r_j, w_j, idx_j = intervals_sorted[j]
            for k in range(1, 5):
                if j >= 1:
                    candidate1 = dp[k][j-1]
                else:
                    candidate1 = (INF_NEG, None)
                
                pos = bisect.bisect_left(R, l_j, 0, j) - 1
                
                if pos < 0:
                    if k == 1:
                        new_tuple = (idx_j,)
                        candidate2 = (w_j, new_tuple)
                    else:
                        candidate2 = (INF_NEG, None)
                else:
                    prev_weight, prev_tuple = dp[k-1][pos]
                    if prev_weight == INF_NEG:
                        candidate2 = (INF_NEG, None)
                    else:
                        merged_tuple = tuple(sorted(prev_tuple + (idx_j,)))
                        candidate2 = (prev_weight + w_j, merged_tuple)
                
                if candidate1[0] > candidate2[0]:
                    dp[k][j] = candidate1
                elif candidate1[0] < candidate2[0]:
                    dp[k][j] = candidate2
                else:
                    if candidate1[1] is None and candidate2[1] is None:
                        dp[k][j] = candidate1
                    elif candidate1[1] is None:
                        dp[k][j] = candidate2
                    elif candidate2[1] is None:
                        dp[k][j] = candidate1
                    else:
                        if candidate1[1] < candidate2[1]:
                            dp[k][j] = candidate1
                        else:
                            dp[k][j] = candidate2
        
        best_weight = INF_NEG
        best_set = None
        for k in range(5):
            weight, tup = dp[k][n-1]
            if weight > best_weight:
                best_weight = weight
                best_set = tup
            elif weight == best_weight and tup is not None and best_set is not None:
                if tup < best_set:
                    best_set = tup
        
        return list(best_set) if best_set is not None else []