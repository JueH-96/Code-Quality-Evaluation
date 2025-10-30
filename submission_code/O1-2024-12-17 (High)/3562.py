from bisect import bisect_left, insort
from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        """
        We want to select up to 4 non-overlapping intervals that maximize
        the total weight. In case of ties in total weight, we return the
        lexicographically smallest list of chosen interval-indices (1-based)
        in ascending order.

        Approach:
          1) Sort intervals by their end coordinate (r). Let n = len(intervals).
             Assign each interval an original 1-based index.
          2) Precompute p[i]: for the i-th interval in the sorted list (1-based),
             p[i] is the largest j < i such that the j-th interval ends strictly
             before the i-th interval starts (i.e. r_j < l_i). We'll find this
             via binary search on the sorted end-coordinates.
          3) Define a DP table dp[i][k] = maximum sum of weights using exactly k
             intervals among the first i in sorted order. Also maintain sol[i][k]
             = the lexicographically best tuple of chosen original indices that
             achieves dp[i][k].
          4) Transitions:
               - skip i-th: take dp[i-1][k]
               - take i-th: dp[p[i]][k-1] + weight_of(i)
             Compare the sums, and if there is a tie, take the lexicographically
             smaller solution.
          5) At the end, we look at dp[n][k] for k=0..4 to find the overall best.
             Among any ties in sum, pick the lexicographically smallest sol[n][k].
          6) Return that solution as a list of indices.
        """

        n = len(intervals)
        # Attach 1-based original index to each interval: (l, r, w, original_idx)
        arr = []
        for i, (l, r, w) in enumerate(intervals):
            arr.append((l, r, w, i+1))

        # Sort by end coordinate (r), then by start (l), then by original index
        arr.sort(key=lambda x: (x[1], x[0], x[3]))

        # Precompute for each sorted interval i (1-based), p[i] = 
        # the largest j < i with arr[j-1].end < arr[i-1].start
        ends = [arr[i][1] for i in range(n)]
        p = [0]*(n+1)
        for i in range(n):
            l_i = arr[i][0]
            # Find first position pos where ends[pos] >= l_i
            pos = bisect_left(ends, l_i)
            x = pos - 1
            # Ensure x < i (0-based), so that j < i in 1-based indexing
            if x >= i:
                x = i - 1
            if x < 0:
                p[i+1] = 0
            else:
                p[i+1] = x + 1  # convert 0-based to 1-based

        # dp[i][k] = max sum using exactly k intervals among the first i
        dp = [[0]*5 for _ in range(n+1)]
        # sol[i][k] = lexicographically smallest tuple of chosen original indices
        sol = [[() for _ in range(5)] for _ in range(n+1)]

        for i in range(1, n+1):
            (l_i, r_i, w_i, idx_i) = arr[i-1]
            for k in range(1, 5):
                # Option 1: skip this interval
                skip_sum = dp[i-1][k]
                skip_sol = sol[i-1][k]

                # Option 2: take this interval
                take_sum = dp[p[i]][k-1] + w_i
                base_sol = sol[p[i]][k-1]
                # Insert the new index while keeping the tuple sorted
                new_list = list(base_sol)
                insort(new_list, idx_i)
                take_sol = tuple(new_list)

                # Choose the better of skip vs. take
                if skip_sum > take_sum:
                    dp[i][k] = skip_sum
                    sol[i][k] = skip_sol
                elif skip_sum < take_sum:
                    dp[i][k] = take_sum
                    sol[i][k] = take_sol
                else:
                    # tie in sum => pick lexicographically smaller solution
                    dp[i][k] = skip_sum
                    sol[i][k] = skip_sol if skip_sol < take_sol else take_sol

        # Among dp[n][0..4], pick the maximum sum, then the lexicographically smallest
        best_sum = 0
        best_sol = ()
        for k in range(5):
            if dp[n][k] > best_sum:
                best_sum = dp[n][k]
                best_sol = sol[n][k]
            elif dp[n][k] == best_sum:
                if sol[n][k] < best_sol:
                    best_sol = sol[n][k]

        return list(best_sol)