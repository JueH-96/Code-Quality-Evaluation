import sys

class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)

        # Helper function to calculate max/min of a potential function
        # val_func maps index i to the contribution of s[i] to the potential
        def solve_potential(s: str, k: int, val_func, maximize: bool) -> int:
            # dp[c] stores the max/min value of the potential after processing s[0...i-1] using exactly c changes
            # Using O(k) space by storing values at the current step i+1 based on values at step i
            dp = [(-sys.float('inf') if maximize else sys.float('inf'))] * (k + 1)

            # Base case: potential value at step 0 (origin) is 0.
            # This state corresponds to having processed s[0...-1] = empty prefix.
            dp[0] = 0

            # Overall extremum initialized with potential at step 0
            overall_extremum = 0

            for i in range(n):
                current_val = val_func(i, s) # Contribution of original s[i]

                # Need a temporary array to store values at step i+1 based on dp (values at step i)
                new_dp = [(-sys.float('inf') if maximize else sys.float('inf'))] * (k + 1)

                # Iterate through changes count at step i
                for c in range(k + 1):
                    if (maximize and dp[c] == -sys.float('inf')) or (not maximize and dp[c] == sys.float('inf')):
                        continue

                    # Option 1: Don't change s[i]. Uses c changes in s[0...i].
                    # These c changes must be in s[0...i-1]. Value at step i is dp[c].
                    # Value at step i+1 = value at step i + original contribution of s[i].
                    next_val_no_change = dp[c] + current_val

                    # Update new_dp[c] with this possibility
                    if maximize:
                        new_dp[c] = max(new_dp[c], next_val_no_change)
                    else:
                        new_dp[c] = min(new_dp[c], next_val_no_change)

                    # Option 2: Change s[i]. Uses c+1 changes in s[0...i].
                    # These c changes must be in s[0...i-1]. Value at step i is dp[c].
                    # Value at step i+1 = value at step i + best possible contribution of s[i] (+1 or -1).
                    if c < k:
                         next_val_with_change = dp[c] + (1 if maximize else -1)

                         # Update new_dp[c+1] with this possibility
                         if maximize:
                             new_dp[c+1] = max(new_dp[c+1], next_val_with_change)
                         else:
                             new_dp[c+1] = min(new_dp[c+1], next_val_with_change)

                # After iterating through all possible changes `c` at step `i`,
                # new_dp contains the max/min values at step `i+1` for each change count `c'`.
                dp = new_dp # Update dp table for the next iteration (values at step i+1)

                # Update overall extremum with values achievable at step i+1
                for c in range(k + 1):
                     if (maximize and dp[c] != -sys.float('inf')) or (not maximize and dp[c] != sys.float('inf')):
                        if maximize:
                             overall_extremum = max(overall_extremum, dp[c])
                        else:
                             overall_extremum = min(overall_extremum, dp[c])

            return overall_extremum

        # Define potential contribution functions
        # For potential v1 = x+y
        def val_v1_func(i, s): return 1 if s[i] in {'N', 'E'} else -1
        # For potential v2 = x-y
        def val_v2_func(i, s): return 1 if s[i] in {'S', 'E'} else -1
        # For potential v3 = -x+y
        def val_v3_func(i, s): return 1 if s[i] in {'N', 'W'} else -1
        # For potential v4 = -x-y
        def val_v4_func(i, s): return 1 if s[i] in {'S', 'W'} else -1


        # Calculate max and min values for each of the four potentials
        # solve_potential calculates the overall max/min of the potential over all steps [0, n] and changes [0, k]

        max_v1 = solve_potential(s, k, val_v1_func, True)
        min_v1 = solve_potential(s, k, val_v1_func, False)
        max_v2 = solve_potential(s, k, val_v2_func, True)
        min_v2 = solve_potential(s, k, val_v2_func, False)
        max_v3 = solve_potential(s, k, val_v3_func, True)
        min_v3 = solve_potential(s, k, val_v3_func, False)
        max_v4 = solve_potential(s, k, val_v4_func, True)
        min_v4 = solve_potential(s, k, val_v4_func, False)

        # The maximum Manhattan distance |x|+|y| is the maximum among the four potential values |v1|, |v2|, |v3|, |v4|.
        # The maximum Manhattan distance over all steps and all valid paths is the maximum absolute value among the extreme potential values reached.
        # The DP finds the max/min value of a potential over all points reachable by a path using <= k changes.
        # If max(v_m) = M, it means there exists *a* path and *a* step where v_m = M.
        # If min(v_m) = m, it means there exists *a* path and *a* step where v_m = m.
        # The maximum |v_m| achieved is max(|M|, |m|).
        # The overall maximum Manhattan distance is the maximum of these maximum |v_m| over all m.

        max_dist = 0
        max_dist = max(max_dist, abs(max_v1))
        max_dist = max(max_dist, abs(min_v1))
        max_dist = max(max_dist, abs(max_v2))
        max_dist = max(max_dist, abs(min_v2))
        max_dist = max(max_dist, abs(max_v3))
        max_dist = max(max_dist, abs(min_v3))
        max_dist = max(max_dist, abs(max_v4))
        max_dist = max(max_dist, abs(min_v4))

        return max_dist