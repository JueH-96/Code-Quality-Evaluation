from typing import List
import bisect # Included as it was part of the thought process, although not strictly used in the final simple DP loop structure

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # 1. Sort the array
        # Sorting is crucial because the power definition (min absolute difference)
        # simplifies to min adjacent difference for sorted subsequences,
        # and the DP relies on processing elements in order.
        nums.sort()
        
        # 2. Collect all unique positive differences between pairs of elements.
        # These differences define the possible values of the minimum power for any subsequence.
        # The minimum power of any subsequence must be one of these differences or 0.
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(nums[j] - nums[i])
                
        # Convert set to sorted list.
        diffs = sorted(list(diffs))
        
        # Keep only positive differences. Minimum power >= 1 means positive differences.
        # Subsequences with power 0 (containing duplicates) contribute 0 to the sum.
        diffs = [d for d in diffs if d > 0]
        
        # Insert 0 at the beginning. This helps define intervals (0, D_1], (D_1, D_2], ..., (D_{M-1}, D_M].
        # Let the sorted unique positive differences be D_1 < D_2 < ... < D_M.
        # Let D_0 = 0. The intervals are (D_j, D_{j+1}] for j=0...M-1.
        # The sum of powers can be calculated as sum_{p=1}^{max_diff} count_ge(p), where count_ge(p)
        # is the number of subsequences of length k with minimum power >= p.
        # This sum is equal to sum_{j=0}^{M-1} (D_{j+1} - D_j) * count_ge(D_{j+1}).
        diffs.insert(0, 0)
        
        total_sum = 0

        # Function to calculate count_ge(min_diff)
        # Counts subsequences of length `length` from sorted array `arr`
        # such that the minimum difference between any two elements is >= min_diff.
        # Uses O(nk) space and O(n^2k) time per call.
        def calculate_count_ge(min_diff, arr, length):
            n_dp = len(arr)
            
            # dp[i][l] = number of subsequences of length `l` using elements from arr[0...i],
            # where arr[i] is the largest element in the subsequence,
            # and the minimum difference between any two elements in the subsequence is >= min_diff.
            # Since arr is sorted and we build subsequences by picking elements in increasing order,
            # the condition on min difference simplifies to min *adjacent* difference >= min_diff.
            dp = [[0] * (length + 1) for _ in range(n_dp)]

            # Base case: subsequences of length 1.
            # Any single element forms a subsequence of length 1.
            # Min difference is not applicable for length 1, or considered infinite.
            for i in range(n_dp):
                dp[i][1] = 1

            # Fill DP table for lengths 2 to k
            for l in range(2, length + 1):
                for i in range(n_dp): # Consider arr[i] as the largest (last picked) element
                    # To form a subsequence of length `l` ending with arr[i],
                    # we must have formed a subsequence of length `l-1` ending at arr[prev_i],
                    # where prev_i < i, and the new difference arr[i] - arr[prev_i] must be >= min_diff.
                    # Also, the subsequence of length `l-1` ending at arr[prev_i] must satisfy the
                    # minimum difference condition within itself (which is guaranteed by the DP state definition).
                    for prev_i in range(i): # Consider arr[prev_i] as the second largest element
                        if arr[i] - arr[prev_i] >= min_diff:
                            # If we can append arr[i] to a valid subsequence of length l-1 ending at arr[prev_i],
                            # the new subsequence of length l ending at arr[i] is also valid.
                            dp[i][l] = (dp[i][l] + dp[prev_i][l-1]) % MOD

            # The total count of subsequences of length `k` with min diff >= min_diff
            # is the sum of dp[i][k] for all possible ending indices i.
            total_count = 0
            for i in range(n_dp):
                total_count = (total_count + dp[i][length]) % MOD
            
            return total_count

        # Iterate through the intervals of minimum power values defined by the unique differences.
        # Let the sorted unique positive differences be D_1 < D_2 < ... < D_M.
        # Let D_0 = 0. The intervals for minimum power values are (D_j, D_{j+1}] for j=0...M-1.
        # The sum of powers = sum_{p=1}^{max_diff} count_ge(p).
        # This sum can be rewritten as sum_{j=0}^{M-1} (D_{j+1} - D_j) * count_ge(D_{j+1}).
        # This is because count_ge(p) is constant for p within the range (D_j, D_{j+1}].
        for j in range(len(diffs) - 1):
            p_lower = diffs[j]
            p_upper = diffs[j+1]
            
            # count_ge(p_upper) is the number of subsequences with min power >= p_upper
            # For powers in (p_lower, p_upper], count_ge is the same as count_ge(p_upper)
            current_count = calculate_count_ge(p_upper, nums, k)
            
            # The difference D_{j+1} - D_j represents the number of distinct integer power values
            # in the range (D_j, D_{j+1}]. Each of these integer values contributes `current_count`
            # to the total sum (via the formula sum_{p=1}^{max_diff} count_ge(p)).
            diff = p_upper - p_lower
            
            # Add the contribution from this interval to the total sum.
            # Ensure modulo arithmetic at each step to prevent overflow.
            term = (diff * current_count) % MOD
            total_sum = (total_sum + term) % MOD

        return total_sum