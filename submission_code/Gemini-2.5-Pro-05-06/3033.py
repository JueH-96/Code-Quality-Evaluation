import math

class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        diff_indices = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                diff_indices.append(i)

        m = len(diff_indices)
        if m == 0:
            return 0
        if m % 2 != 0:
            return -1

        # dp[k] will be the minimum cost to fix the first k differences
        # (i.e., diff_indices[0]...diff_indices[k-1])
        # k is the count of differences, so dp array size is m+1.
        # dp indices are 0, 2, 4, ..., m.
        dp = [math.inf] * (m + 1)
        dp[0] = 0

        # The outer loop `i` goes from 2 to m (step 2).
        # `i` represents the count of differences being considered: P[0]...P[i-1].
        for i in range(2, m + 1, 2):
            # Option 1: Pair P[i-2] and P[i-1] (using 0-indexed diff_indices).
            # Cost for this pair is min(x, diff_indices[i-1] - diff_indices[i-2]).
            # This is added to dp[i-2] (cost to fix P[0]...P[i-3]).
            cost_pair_adj = min(x, diff_indices[i-1] - diff_indices[i-2])
            if dp[i-2] != math.inf:
                 dp[i] = min(dp[i], dp[i-2] + cost_pair_adj)
            
            # Option 2: Pair P[j_arr_idx] and P[i-1] using operation type 1 (cost x).
            # `j_arr_idx` is the array index of the first element of the X-pair.
            # The elements P[0]...P[j_arr_idx-1] are fixed with cost dp[j_arr_idx].
            # (dp state takes count, so dp[j_arr_idx] refers to first j_arr_idx elements, which are P[0]..P[j_arr_idx-1])
            # The elements P[j_arr_idx+1]...P[i-2] must be fixed by pairing adjacent items.
            # `current_adj_pair_cost_sum` stores sum of (P[k+1]-P[k]) for these intermediate elements.
            
            current_adj_pair_cost_sum = 0
            # Loop `j_arr_idx` from `i-2` down to `0` (step -2).
            # `j_arr_idx` is the actual index in `diff_indices`.
            for j_arr_idx in range(i - 2, -1, -2):
                # If j_arr_idx == i-2: The interval P[j_arr_idx+1]...P[i-2] is empty. Sum = 0.
                # This is correctly handled by initializing current_adj_pair_cost_sum = 0
                # and only adding to it if j_arr_idx < i-2.

                # If j_arr_idx < i-2: This means j_arr_idx was previously j_arr_idx+2.
                # The interval of adjacently-paired elements expanded to the left
                # by adding the pair (P[j_arr_idx+1], P[j_arr_idx+2]).
                # So, add (diff_indices[j_arr_idx+2] - diff_indices[j_arr_idx+1]) to the sum.
                if j_arr_idx + 1 < i - 1 : # Check if P[j_arr_idx+1] and P[j_arr_idx+2] exist
                    # This check ensures j_arr_idx+2 <= i-2, which means elements for this pair exist.
                    current_adj_pair_cost_sum += (diff_indices[j_arr_idx+2] - diff_indices[j_arr_idx+1])
                
                # Cost for this configuration:
                # dp[j_arr_idx]: cost to fix P[0]...P[j_arr_idx-1]
                # x: cost to pair P[j_arr_idx] and P[i-1]
                # current_adj_pair_cost_sum: cost for P[j_arr_idx+1]...P[i-2]
                if dp[j_arr_idx] != math.inf:
                    dp[i] = min(dp[i], dp[j_arr_idx] + x + current_adj_pair_cost_sum)

        if dp[m] == math.inf: # Should not happen if m is even and >0, unless costs are huge (not an issue here)
            return -1 # This case is mainly for m being odd, already handled.
                      # Or if somehow no path is found, though one always exists if m is even.
        
        return dp[m]