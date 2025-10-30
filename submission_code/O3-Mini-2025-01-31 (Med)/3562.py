from bisect import bisect_right
from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        # Enrich intervals with their original index: (l, r, weight, orig)
        enriched = []
        for idx, (l, r, w) in enumerate(intervals):
            enriched.append((l, r, w, idx))
        # Sort by starting point; if tie, by original index (to stabilize binary search and lex orders)
        enriched.sort(key=lambda x: (x[0], x[3]))
        
        # Extract the list of starting points for binary search.
        starts = [x[0] for x in enriched]
        
        # We'll allow up to 4 intervals. Let dp[i][k] store a tuple: (best_sum, best_sequence)
        # where best_sequence is a list of original indices.
        # dp[i][k] means the best answer (starting from index i in enriched) when you can choose up to k intervals.
        # Dimensions: dp[0..n][0..4] since k goes from 0 to 4.
        dp = [[(0, []) for _ in range(5)] for _ in range(n+1)]
        
        # Base: for any state dp[n][k] = (0, []) and dp[i][0] = (0, []) because if we cannot pick any intervals then weight=0.
        for i in range(n-1, -1, -1):
            for rem in range(5):
                if rem == 0:
                    dp[i][rem] = (0, [])
                else:
                    # Option 1: skip the interval at i.
                    skip_weight, skip_seq = dp[i+1][rem]
                    
                    # Option 2: choose interval i.
                    l, r, w, orig = enriched[i]
                    # find the next interval with start > r (since sharing a boundary is not allowed)
                    next_i = bisect_right(starts, r)
                    pick_weight = w + dp[next_i][rem-1][0]
                    pick_seq = [orig] + dp[next_i][rem-1][1]
                    
                    # Pick the option with greater weight sum.
                    # In case of a tie, pick the lexicographically smallest sequence.
                    if pick_weight > skip_weight:
                        best = (pick_weight, pick_seq)
                    elif pick_weight < skip_weight:
                        best = (skip_weight, skip_seq)
                    else:
                        # When both choices have same total weight, choose lexicographically smallest sequence.
                        best = (pick_weight, pick_seq) if pick_seq < skip_seq else (skip_weight, skip_seq)
                    dp[i][rem] = best
        
        # The answer is the best result we can get starting from index 0 when allowed to choose at most 4 intervals.
        return dp[0][4][1]