from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Build prefix sum array P, where P[i] = sum(nums[0:i])
        P = [0] * (n + 1)
        for i in range(1, n + 1):
            P[i] = P[i - 1] + nums[i - 1]
        
        # Explanation:
        # We want to choose k disjoint contiguous subarrays in order.
        # Their "strength" is defined as:
        #    strength = sum[1]*k − sum[2]*(k − 1) + sum[3]*(k − 2) − ... + (−1)^(k+1)*sum[k]*1.
        #
        # Notice that the multiplier for the j-th chosen subarray (1-indexed)
        # is: w(j) = (–1)^(j+1) * (k − j + 1). Equivalently, if we count j from 0
        # (so that the next chosen subarray is the (j+1)-th) then the multiplier is:
        #    m = (–1)^j * (k − j).
        #
        # Let dp[j][i] be the maximum strength we can get from the first i elements
        # by choosing j subarrays (j from 0 to k). With no subarrays chosen we have dp[0][i] = 0.
        # To “cut” a new subarray ending at some index t (with t between i+1 and n),
        # its contribution would be m * (P[t] − P[i]), and we add dp[j][i] (the best we could do up to i).
        #
        # That is, for j from 0 to k − 1 and for t from 1 to n one may write:
        #   dp[j+1][t] = max_{0 ≤ i < t} (dp[j][i] + m * (P[t] − P[i])).
        #
        # Rearranging:
        #   dp[j+1][t] = m * P[t] + max_{0 ≤ i < t} (dp[j][i] - m * P[i]).
        # We then “propagate” the best value in the dp[j+1] row so that
        # dp[j+1][t] = max(dp[j+1][t-1], dp[j+1][t])
        # This propagation is like giving the option to “not use” the t-th element.
        #
        # The final answer is dp[k][n].
        
        # We'll use 1D arrays for each stage j.
        NEG_INF = -10**30  # a large negative number
        
        # For j = 0 (no subarrays chosen) we can always have 0 strength,
        # regardless of how many elements we have processed.
        dp_prev = [0] * (n + 1)
        
        # Now build dp for each subarray count from 1 to k.
        for j in range(0, k):  # j is the number of subarrays chosen so far.
            # The multiplier for the next (j+1)-th interval:
            m = ((-1) ** j) * (k - j)
            dp_curr = [NEG_INF] * (n + 1)
            dp_curr[0] = NEG_INF  # cannot form a non-empty subarray from 0 elements
            best = NEG_INF
            # We want to compute for each t (1 <= t <= n):
            #    candidate = (max_{i in [0,t)} (dp_prev[i] - m * P[i])) + m * P[t]
            # and then make dp_curr[t] be the maximum value seen so far.
            for t in range(1, n + 1):
                # Update best value among indices [0, t-1]
                candidate_index = dp_prev[t - 1] - m * P[t - 1]
                if candidate_index > best:
                    best = candidate_index
                candidate_value = best + m * P[t]
                # Propagate the best so far so that dp_curr is non-decreasing
                if t == 1:
                    dp_curr[t] = candidate_value
                else:
                    dp_curr[t] = dp_curr[t - 1] if dp_curr[t - 1] > candidate_value else candidate_value
            dp_prev = dp_curr  # move to next stage
        
        return dp_prev[n]

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.maximumStrength([1, 2, 3, -1, 2], 3))   # Expected output: 22
    # Example 2:
    print(sol.maximumStrength([12, -2, -2, -2, -2], 5)) # Expected output: 64
    # Example 3:
    print(sol.maximumStrength([-1, -2, -3], 1))         # Expected output: -1