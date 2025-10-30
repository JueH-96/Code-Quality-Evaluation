from typing import List
import bisect

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # ----------------------------------------------------------------------
        # We want to choose up to 4 non-overlapping intervals that maximize the
        # total weight. Among all optimal solutions, we return the lexicographically
        # smallest list of (1-based) indices.
        #
        # Approach:
        #  1. Sort intervals by their right boundary r.
        #  2. For each interval i, find p[i], the largest j < i such that
        #     intervals[j].r < intervals[i].l (strict inequality for non-overlap).
        #  3. Use dynamic programming with dimension dp[i][k] for i in [0..n]
        #     and k in [0..4], where dp[i][k] holds:
        #       (best_score, chosen_indices_list (up to k intervals))
        #     We carefully break ties by lexicographic order of chosen_indices_list.
        #  4. The answer is the best among dp[n][k] for k=0..4.
        #
        # Time complexity: O(n log n) due to sorting + O(n log n) for finding p[]
        #                  + O(n * 4) for the DP (each DP step is O(1) to O(k) = O(4)
        #                  for tie-breaking, which is acceptable).
        # ----------------------------------------------------------------------

        n = len(intervals)
        if n == 0:
            return []

        # Attach original indices (1-based) and sort by r
        # intervals[i] = [l, r, weight]
        enumerated_intervals = []
        for i, (l, r, w) in enumerate(intervals):
            enumerated_intervals.append((l, r, w, i+1))  # store (l, r, w, original_idx)

        enumerated_intervals.sort(key=lambda x: x[1])  # sort by r

        # Precompute p[i]: for the i-th interval in sorted order,
        # find the largest j < i where intervals[j].r < intervals[i].l
        # We'll use binary search on the array of end-times.
        ends = [interval[1] for interval in enumerated_intervals]  # sorted by r
        p = [-1] * n
        for i in range(n):
            l_i = enumerated_intervals[i][0]
            # find rightmost j where ends[j] < l_i
            # bisect_left(ends, l_i) gives the leftmost index with ends[idx] >= l_i
            # so the predecessor is bisect_left(...)-1
            j = bisect.bisect_left(ends, l_i) - 1
            p[i] = j

        # We'll store in dp_score[i][k] the maximum total weight using intervals
        # up to index i-1 (in sorted order), with k intervals chosen.
        # We'll store in dp_choice[i][k] the chosen original indices (sorted) that
        # achieve dp_score[i][k].
        dp_score = [[0]*5 for _ in range(n+1)]
        dp_choice = [[[] for _ in range(5)] for _ in range(n+1)]

        # Helper to compare two lists a, b lexicographically
        def compare_lexi(a: List[int], b: List[int]) -> int:
            # return -1 if a < b lex, 1 if a > b, 0 if equal
            for x, y in zip(a, b):
                if x < y:
                    return -1
                elif x > y:
                    return 1
            if len(a) < len(b):
                return -1
            elif len(a) > len(b):
                return 1
            return 0

        # Main DP
        for i in range(1, n+1):
            # interval i-1 in sorted order has (l, r, w, original_idx)
            l_i, r_i, w_i, orig_i = enumerated_intervals[i-1]
            for k in range(5):
                # Option A: don't take interval i-1
                dp_score[i][k] = dp_score[i-1][k]
                dp_choice[i][k] = dp_choice[i-1][k]

                # Option B: take interval i-1 (if k > 0)
                if k > 0:
                    # predecessor
                    j = p[i-1]  # p is 0-based, i-1 is 0-based
                    if j == -1:
                        prev_score = 0
                        prev_choice = []
                    else:
                        prev_score = dp_score[j+1][k-1]
                        prev_choice = dp_choice[j+1][k-1]

                    candidate_score = prev_score + w_i
                    # build candidate choice (sorted by original index)
                    candidate_choice = prev_choice[:]
                    candidate_choice.append(orig_i)
                    candidate_choice.sort()

                    if candidate_score > dp_score[i][k]:
                        dp_score[i][k] = candidate_score
                        dp_choice[i][k] = candidate_choice
                    elif candidate_score == dp_score[i][k]:
                        # tie-break by lexicographic order
                        if compare_lexi(candidate_choice, dp_choice[i][k]) < 0:
                            dp_choice[i][k] = candidate_choice

        # Find the best among dp_score[n][k], k=0..4
        best_score = 0
        best_indices = []
        for k in range(5):
            s = dp_score[n][k]
            if s > best_score:
                best_score = s
                best_indices = dp_choice[n][k]
            elif s == best_score:
                # tie-break
                if compare_lexi(dp_choice[n][k], best_indices) < 0:
                    best_indices = dp_choice[n][k]

        # best_indices is already sorted to satisfy the lexicographically smallest rule
        return best_indices