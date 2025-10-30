from typing import List
import sys

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # dp[j][i][0]: max sum using first i elements to form j non-overlapping subarrays,
        # where the j-th subarray does NOT end at index i-1.
        # dp[j][i][1]: max sum using first i elements to form j non-overlapping subarrays,
        # where the j-th subarray DOES end at index i-1.
        # Dimensions: (k+1) x (n+1) x 2
        
        # Use a large negative number to represent unreachable states.
        # Sufficiently small value given the constraints on nums[i] and n.
        # Max sum is around n * 10^4, min sum is around n * -10^4.
        # Intermediate values max(dp) - P are around 2 * n * 10^4.
        # Resulting dp values are around n * 10^4.
        # -sys.maxsize // 2 is a safe large negative number.
        NEG_INF = -sys.maxsize // 2 
        
        dp = [[[NEG_INF for _ in range(2)] for _ in range(n + 1)] for _ in range(k + 1)]
        
        # Prefix sums
        # P[i] = sum(nums[0...i-1])
        P = [0] * (n + 1)
        for i in range(n):
            P[i+1] = P[i] + nums[i]
            
        # Base case: 0 subarrays, sum is 0, using any number of elements.
        # It's possible to form 0 subarrays with sum 0 using the first i elements.
        # By definition, the 0-th subarray cannot "end" anywhere.
        for i in range(n + 1):
            dp[0][i][0] = 0

        # Iterate through the number of subarrays j (from 1 to k)
        for j in range(1, k + 1):
            # max_prev_dp_minus_P: This variable will store the maximum value of
            # (max(dp[j-1][p][0], dp[j-1][p][1]) - P[p]) for valid previous ending positions p.
            # The valid range for p is (j-1)*m <= p <= i-m.
            # We update this value iteratively as i increases.
            max_prev_dp_minus_P = NEG_INF

            # Iterate through the number of elements considered i (from 1 to n)
            # i corresponds to using elements nums[0] to nums[i-1].
            for i in range(1, n + 1):
                # Calculate dp[j][i][0]: max sum using first i elements, j subarrays, last one not ending at i-1.
                # This means the j subarrays must use only the first i-1 elements.
                # The max sum is the max sum using first i-1 elements with j subarrays,
                # where the last subarray either ended at i-2 (dp[j][i-1][1]) or before i-2 (dp[j][i-1][0]).
                dp[j][i][0] = max(dp[j][i-1][0], dp[j][i-1][1])

                # Calculate dp[j][i][1]: max sum using first i elements, j subarrays, last one ending at i-1.
                # This j-th subarray ends at index i-1 and must have length >= m.
                # It must start at some index p where (i-1) - p + 1 >= m => p <= i - m.
                # The elements used for the previous j-1 subarrays are nums[0...p-1], which corresponds to P[p].
                # The maximum sum for the previous j-1 subarrays using these elements is max(dp[j-1][p][0], dp[j-1][p][1]).
                # The total sum is max(dp[j-1][p][0], dp[j-1][p][1]) + sum(nums[p...i-1]) = max(...) + P[i] - P[p].
                # We need to maximize this over all valid starting positions p.
                # A valid previous state dp[j-1][p] requires using at least (j-1)*m elements, so p >= (j-1)*m.
                # Thus, the valid range for p is (j-1)*m <= p <= i-m.
                # The state dp[j][i][1] is only possible if we have used at least j*m elements in total (i >= j*m).

                # Check if the index `i-m` is a valid potential starting position `p` for the current subarray
                # that allows space for the previous `j-1` subarrays.
                # This means `i-m` must be >= (j-1)*m. (Also implies i-m >= 0 since j>=1, m>=1)
                # This check simplifies to `i >= j*m`.
                if i >= j * m:
                    # When calculating dp[j][i][1], we need the maximum of (max(dp[j-1][p][0], dp[j-1][p][1]) - P[p])
                    # over the range (j-1)*m <= p <= i-m.
                    # As we iterate i, the upper bound `i-m` increases.
                    # The new potential value for the maximum comes from the index p = i-m.
                    
                    # Get the max sum using first i-m elements with j-1 subarrays.
                    val_at_i_minus_m = max(dp[j-1][i-m][0], dp[j-1][i-m][1])
                    
                    # Update the running maximum needed for calculating dp[j][i][1].
                    # This running maximum `max_prev_dp_minus_P` represents
                    # max_{ (j-1)*m <= p' <= i-m } (max(dp[j-1][p'][0], dp[j-1][p'][1]) - P[p'])
                    # The value at p = i-m is max(dp[j-1][i-m][0], dp[j-1][i-m][1]) - P[i-m].
                    if val_at_i_minus_m != NEG_INF:
                         max_prev_dp_minus_P = max(max_prev_dp_minus_P, val_at_i_minus_m - P[i-m])

                    # If we have found at least one valid previous state (i.e., max_prev_dp_minus_P is not NEG_INF)
                    # and we have enough elements to form j subarrays ending at i-1 (i >= j*m).
                    if max_prev_dp_minus_P != NEG_INF:
                        # The maximum sum ending the j-th subarray at i-1 is max_prev_dp_minus_P + P[i].
                        dp[j][i][1] = max_prev_dp_minus_P + P[i]
                    # else: dp[j][i][1] remains NEG_INF

        # The final answer is the maximum sum using first n elements forming exactly k subarrays,
        # where the last subarray either ends at index n-1 (state dp[k][n][1]) or earlier (state dp[k][n][0]).
        result = max(dp[k][n][0], dp[k][n][1])
        
        # According to constraints (1 <= k <= floor(n/m)), it is always possible to form k subarrays.
        # Thus, the result should not be NEG_INF.
        return result