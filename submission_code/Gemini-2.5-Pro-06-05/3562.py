import bisect
from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        if n == 0:
            return []

        # Augment with original indices.
        indexed_intervals = []
        for i in range(n):
            indexed_intervals.append(intervals[i] + [i])

        # Sort by start time. Python's sort is stable, which is good for
        # deterministic behavior when start times are equal.
        indexed_intervals.sort(key=lambda x: x[0])

        # Create a list of start times for efficient binary search.
        starts = [iv[0] for iv in indexed_intervals]

        # dp[i][k] stores a tuple: (max_score, indices_list) for choosing
        # exactly k non-overlapping intervals from the suffix starting at index i.
        # Initialize with a score of -1 to indicate impossibility.
        dp = [[(-1, []) for _ in range(5)] for _ in range(n + 1)]

        # Base case: for any suffix, choosing 0 intervals gives a score of 0.
        for i in range(n + 1):
            dp[i][0] = (0, [])

        # Helper to compare two (score, indices) pairs.
        # It prioritizes higher score, then lexicographically smaller indices.
        def get_better_pair(p1, p2):
            s1, l1 = p1
            s2, l2 = p2
            if s1 > s2:
                return p1
            if s2 > s1:
                return p2
            # Scores are equal, choose the lexicographically smaller list.
            if l1 < l2:
                return p1
            else:
                return p2

        # Fill the DP table from right to left.
        for i in range(n - 1, -1, -1):
            _l_i, r_i, w_i, original_idx_i = indexed_intervals[i]

            # Find the index 'j' of the first interval that starts after interval 'i' ends.
            # An interval is non-overlapping if its start is strictly greater than
            # the other's end. This means start_j >= r_i + 1.
            j = bisect.bisect_left(starts, r_i + 1, lo=i + 1)

            for k in range(1, 5):
                # Option 1: Skip interval 'i'.
                pair_skip = dp[i + 1][k]

                # Option 2: Pick interval 'i'.
                pair_pick = (-1, [])
                prev_score, prev_indices = dp[j][k - 1]
                if prev_score != -1:
                    score_pick = w_i + prev_score
                    # Keep the indices list sorted for lexicographical comparison.
                    # As len(prev_indices) <= 3, this is very fast.
                    indices_pick = sorted([original_idx_i] + prev_indices)
                    pair_pick = (score_pick, indices_pick)

                # dp[i][k] is the better of the two options.
                dp[i][k] = get_better_pair(pair_skip, pair_pick)

        # The final answer can have 1, 2, 3, or 4 intervals.
        # Find the best result among all these possibilities.
        best_pair = (0, [])
        for k in range(1, 5):
            best_pair = get_better_pair(best_pair, dp[0][k])

        return best_pair[1]