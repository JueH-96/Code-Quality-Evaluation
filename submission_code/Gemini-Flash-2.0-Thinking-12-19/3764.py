from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        n = len(grid)
        # m = len(grid[0]) # m is implicitly handled by limits[i] <= m and sorting row

        # Precompute prefix sums of sorted rows
        # row_prefix_sums[i][c] will store the sum of the largest c elements in grid[i]
        row_prefix_sums = []
        for row in grid:
            # Sort each row in descending order
            sorted_row = sorted(row, reverse=True)
            
            # Calculate prefix sums
            # current_prefix_sum[c] is the sum of the top c elements (0-indexed from sorted list)
            # Size will be len(row) + 1 = m + 1
            current_prefix_sum = [0] # Sum of 0 elements is 0
            current_sum = 0
            for val in sorted_row:
                current_sum += val
                current_prefix_sum.append(current_sum)
            row_prefix_sums.append(current_prefix_sum)

        # DP table: dp[s] is the maximum sum using exactly s elements
        # from the rows processed so far (rows 0 to i).
        # Initialize with -float('inf') to indicate unreachable states, except dp[0].
        # Since grid values are non-negative, a sum of 0 is always possible by taking 0 items.
        dp = [-float('inf')] * (k + 1)
        dp[0] = 0 # Sum is 0 when 0 elements are selected initially

        # Iterate through each row
        for i in range(n):
            limit_i = limits[i]
            current_row_psum = row_prefix_sums[i]
            
            # Update dp table using elements from the current row (row i)
            # We want to calculate dp[s] using dp[s-c] from the previous row.
            # To avoid using the current row's contribution multiple times for the same s
            # in the same row iteration, iterate s downwards from k to 0.
            # s is the target total number of elements selected (from row 0 to i).
            for s in range(k, -1, -1):
                # Consider taking c elements from the current row (row i).
                # c can range from 1 up to min(s, limit_i).
                # Also, we can only take up to the number of elements available in the row.
                # Since limits[i] <= m = len(grid[i]), min(s, limit_i) is sufficient.
                max_c = min(s, limit_i)

                # Iterate through possible counts 'c' from the current row
                # We start c from 1 because the c=0 case (taking 0 from current row)
                # is implicitly covered by the fact that dp[s] is initialized with
                # the maximum sum for 's' elements from the previous rows.
                for c in range(1, max_c + 1):
                    prev_s = s - c # Number of elements that must come from previous rows (0 to i-1)
                    
                    # Check if the state dp[prev_s] (max sum with prev_s elements from rows 0 to i-1)
                    # was reachable. If dp[prev_s] is -inf, it means it's impossible to get
                    # exactly prev_s elements from the previous rows.
                    if dp[prev_s] != -float('inf'):
                        # Update dp[s] with the possibility of taking c elements from row i
                        # combined with the best sum for prev_s elements from previous rows.
                        # current_row_psum[c] is the sum of the largest c elements from grid[i].
                        dp[s] = max(dp[s], dp[prev_s] + current_row_psum[c])

        # The answer is the maximum value in the dp table from index 0 to k.
        # max(dp) finds the maximum sum achievable using exactly s elements, for 0 <= s <= k.
        # Since dp[0] is 0, and grid values are non-negative, max(dp) will be at least 0.
        # If k=0, dp contains only dp[0]=0 and others -inf, max returns 0.
        # If k>0 but no elements can be picked (e.g., all limits are 0), only dp[0] remains 0, max returns 0.
        return max(dp)