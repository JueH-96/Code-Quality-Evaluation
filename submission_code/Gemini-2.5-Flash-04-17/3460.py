from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        MAX_INV = 400 # Maximum possible cnt_i based on constraints

        # Map required length (end_i + 1) to required inversion count (cnt_i)
        req_map = {req[0] + 1: req[1] for req in requirements}

        # dp[j] will store the number of permutations of [0, ..., i-1] with j inversions
        # that satisfy all requirements up to length i.
        # We use 1D DP because dp[i] only depends on dp[i-1].
        # The size is MAX_INV + 1 as inversion counts are bounded by 400.
        dp = [0] * (MAX_INV + 1)

        # Base case: permutation of [] (length 0) has 0 inversions.
        dp[0] = 1

        # Iterate through prefix lengths i from 1 to n
        for i in range(1, n + 1):
            # Calculate prefix sums for dp from the previous step (representing dp[i-1])
            prefix_sum = [0] * (MAX_INV + 1)
            prefix_sum[0] = dp[0]
            for j in range(1, MAX_INV + 1):
                prefix_sum[j] = (prefix_sum[j-1] + dp[j]) % MOD

            # Calculate new_dp for length i
            new_dp = [0] * (MAX_INV + 1)
            
            # j is the current inversion count for prefix length i
            for j in range(MAX_INV + 1):
                # A permutation of [0, ..., i-1] with j inversions is formed by
                # inserting (i-1) into a permutation of [0, ..., i-2] with k inversions.
                # Inserting (i-1) at position p (0-indexed, 0 <= p <= i-1) adds (i-1) - p new inversions.
                # Let 'a' be the number of added inversions, a = (i-1) - p, so 0 <= a <= i-1.
                # The new total inversion count is j = k + a, so k = j - a.
                # We sum dp[i-1][k] for k = j - a, where 0 <= a <= i-1 and k >= 0.
                # This is sum dp[i-1][j - a] for a in [0, i-1], or sum dp[i-1][k] for k in [j - (i-1), j].
                # We also need k >= 0, so k is in [max(0, j - (i-1)), j].

                lower_bound_k = max(0, j - (i-1))
                upper_bound_k = j # Sum up to j from previous step dp[i-1]

                # The sum of dp[i-1][k] for k from lower_bound_k to upper_bound_k
                # using prefix sums: S[upper_bound_k] - S[lower_bound_k - 1]
                sum_dp_prev = (prefix_sum[upper_bound_k] - (prefix_sum[lower_bound_k - 1] if lower_bound_k > 0 else 0) + MOD) % MOD
                
                new_dp[j] = sum_dp_prev
                
            # Apply requirement for length i, if any
            if i in req_map:
                required_cnt = req_map[i]
                
                # The maximum possible inversions for a permutation of length i is i * (i - 1) // 2.
                # If the required_cnt exceeds this theoretical maximum, then no permutation of length i
                # can satisfy this requirement.
                max_possible_inv_at_i = i * (i - 1) // 2
                
                if required_cnt > max_possible_inv_at_i:
                     # This requirement is impossible. The number of permutations for this length
                     # with the required count is 0. Since all other counts for this length
                     # must also be 0 according to the requirement, the entire new_dp should become zeros.
                     new_dp = [0] * (MAX_INV + 1)
                else:
                     # The required count is possible. Zero out all counts that don't match.
                     temp_new_dp = [0] * (MAX_INV + 1)
                     # required_cnt is guaranteed to be <= MAX_INV by problem constraints
                     temp_new_dp[required_cnt] = new_dp[required_cnt]
                     new_dp = temp_new_dp

            # Update dp to be new_dp for the next iteration
            dp = new_dp

        # The final answer is the number of permutations of length n (i.e., the full permutation)
        # with the required inversion count for end_i = n - 1.
        # Find the requirement for end_i = n - 1.
        final_cnt = -1
        for req in requirements:
            if req[0] == n - 1:
                final_cnt = req[1]
                break

        # The constraints guarantee that there is at least one i such that end_i == n-1.
        # The constraints also guarantee 0 <= cnt_i <= 400.
        # So final_cnt should be found and be <= MAX_INV.
        # Return dp[final_cnt]
        if final_cnt == -1: # Should not happen based on constraints
            return 0
            
        return dp[final_cnt]