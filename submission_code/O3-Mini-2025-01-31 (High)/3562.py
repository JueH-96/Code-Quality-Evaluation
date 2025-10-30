from typing import List
import bisect

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        # We are allowed to choose at most 4 intervals.
        maxCount = 4
        # A very small number to denote an impossible state.
        NEG = -10**15

        # Build a list of intervals with their original index.
        # Each element is a tuple: (l, r, weight, orig_index)
        # We sort primarily by r, then by l and then by original index.
        sorted_intervals = sorted(
            [(l, r, w, i) for i, (l, r, w) in enumerate(intervals)],
            key = lambda x: (x[1], x[0], x[3])
        )
        
        # Precompute the list of end times in sorted order.
        ends = [interval[1] for interval in sorted_intervals]
        # For each interval in sorted_intervals, find the largest index j such that
        # sorted_intervals[j].r < current interval's l. (Strictly less because sharing
        # a boundary is considered overlapping.)
        prev = []
        for i, (l, r, w, orig) in enumerate(sorted_intervals):
            # bisect_left finds the first position where l could be inserted to keep sorted order.
            # Subtract one to get the last interval with end < l.
            j = bisect.bisect_left(ends, l) - 1
            prev.append(j)
        
        # dp[i][k] will represent the best achievable (score, combination)
        # using the first i intervals in sorted_intervals (i.e. indices 0..i-1)
        # with exactly k intervals chosen.
        # The "combination" is stored as a tuple of original indices sorted in ascending order.
        # We want to maximize the score; if scores tie, we choose the lexicographically
        # smallest combination.
        # We'll build a dp table of dimensions (n+1) x (maxCount+1).
        dp = [[None] * (maxCount+1) for _ in range(n+1)]
        
        # Base case: If no intervals are considered, the only feasible selection is k==0.
        dp[0][0] = (0, ())
        for k in range(1, maxCount+1):
            dp[0][k] = (NEG, ())
        
        # Fill dp table.
        for i in range(1, n+1):
            # current interval (in sorted order) is at index i-1.
            l, r, w, orig = sorted_intervals[i-1]
            # For k==0, we cannot pick any interval, so just carry over.
            dp[i][0] = dp[i-1][0]
            for k in range(1, maxCount+1):
                # Option 1: Skip the current interval.
                candidate_skip = dp[i-1][k]
                
                # Option 2: Take the current interval if possible.
                # We have to add the current interval’s weight to the best state
                # that ended before this interval starts.
                j = prev[i-1]  # j is the index (in sorted_intervals) of the last non overlapping interval.
                # dp[j+1][k-1] represents the best state using intervals 0..j with k-1 picks.
                base_score, base_comb = dp[j+1][k-1]
                candidate_take = (base_score + w, tuple(sorted(base_comb + (orig,))))
                
                # Now choose whichever candidate is better.
                # Our priority is: higher score; if equal score, then lexicographically smaller combination.
                if candidate_take[0] > candidate_skip[0]:
                    best_candidate = candidate_take
                elif candidate_take[0] < candidate_skip[0]:
                    best_candidate = candidate_skip
                else:
                    # Scores are equal. Lexicographical comparison of combinations (which are sorted tuples)
                    if candidate_take[1] < candidate_skip[1]:
                        best_candidate = candidate_take
                    else:
                        best_candidate = candidate_skip
                dp[i][k] = best_candidate
        
        # Among all states with up to 4 intervals chosen, pick the best one.
        best_state = (0, ())
        for k in range(maxCount+1):
            state = dp[n][k]
            if state[0] > best_state[0]:
                best_state = state
            elif state[0] == best_state[0]:
                if state[1] < best_state[1]:
                    best_state = state
        # The chosen combination is stored as a sorted tuple of original indices.
        return list(best_state[1])