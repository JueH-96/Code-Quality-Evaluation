from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        """
        We use a dynamic programming (DP) approach, where dp[j][i] represents the maximum
        strength achievable using j subarrays from the first i elements of nums.
        
        Let prefix[i] = sum of nums[0..i-1].
        
        Define the coefficient a_j for the j-th subarray (1-based) as:
            a_j = ((-1)^(j+1)) * (k - j + 1)
        i.e. for j = 1..k:
           if j is odd:   a_j = +(k - j + 1)
           if j is even:  a_j = -(k - j + 1)

        Recurrence:
          dp[j][i] = max(
              dp[j][i-1],      # skip the i-th element
              a_j * prefix[i] + max_{0 <= p < i}[ dp[j-1][p] - a_j * prefix[p] ]
          )
        
        To make this O(n*k) (where n*k <= 10^6), we keep track of:
          best_j(i) = max_{0 <= p <= i} [ dp[j-1][p] - a_j * prefix[p] ]
        in a running fashion so we don't have to scan for p each time.

        The final answer will be dp[k][n].
        """

        n = len(nums)

        # Compute prefix sums: prefix[i] = sum of nums[0..i-1].
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        # We'll use rolling arrays to store dp for memory efficiency.
        # dp_prev[i] will represent dp[j-1][i], dp_curr[i] will represent dp[j][i].
        dp_prev = [0] * (n + 1)
        # For j>0, dp[j][0] should be -inf, so we set them after the first iteration starts.

        # Pre-fill dp_prev for j=0: picking 0 subarrays => strength = 0.
        # dp_prev = [0, 0, ..., 0]

        # Iterate for each subarray index j = 1..k
        for j in range(1, k + 1):
            # Compute the coefficient a_j = ((-1)^(j+1)) * (k - j + 1)
            sign = 1 if (j % 2) == 1 else -1
            a_j = sign * (k - j + 1)

            dp_curr = [float('-inf')] * (n + 1)
            best = [float('-inf')] * (n + 1)

            # Initialize best[0]
            # best[0] = dp_prev[0] - a_j * prefix[0] = dp_prev[0] - a_j*0
            # dp_prev[0] is 0 only if j-1=0, else -inf
            best[0] = dp_prev[0] - a_j * prefix[0]

            # Fill for i = 1..n
            for i in range(1, n + 1):
                # dp_curr[i] can be the same as dp_curr[i-1] if we skip the i-th element,
                # or we take subarray j ending at i-1 (coefficient * sum(...) + best[i-1]).
                dp_curr[i] = max(
                    dp_curr[i - 1],
                    a_j * prefix[i] + best[i - 1]
                )
                # Update best[i] for use in future i' > i
                best[i] = max(
                    best[i - 1],
                    dp_prev[i] - a_j * prefix[i]
                )

            dp_prev = dp_curr  # Move current DP row to previous for next j

        # The result is dp[k][n], stored now in dp_prev[n]
        return dp_prev[n]