import bisect
from typing import List

class Solution:
    """
    Calculates the sum of powers of all subsequences of nums with length k.
    The power of a subsequence is defined as the minimum absolute difference 
    between any two elements in the subsequence.
    The sum is returned modulo 10^9 + 7.

    The approach uses the property that the sum of powers can be computed as:
    Sum = sum_{d=1 to max_possible_difference} N(>= d)
    where N(>= d) is the number of subsequences of length k whose power is at least d.

    This sum can be further simplified by considering only distinct positive differences
    d_1 < d_2 < ... < d_m that can occur between elements of nums. Let d_0 = 0.
    Sum = sum_{i=1 to m} (d_i - d_{i-1}) * N(>= d_i)

    N(>= d) is computed using dynamic programming with prefix sums for optimization.
    The DP state dp[i][j] counts subsequences of length j ending at nums[i] with minimum difference >= d.
    Prefix sums allow calculating the sum of dp[p][j-1] for relevant p efficiently using binary search.
    """
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        """
        :param nums: List[int], the input array of integers.
        :param k: int, the required length of subsequences.
        :return: int, the sum of powers modulo 10^9 + 7.
        """
        
        n = len(nums)
        # Sort the input array. This is crucial for the DP approach and for calculating differences.
        nums.sort()
        
        MOD = 10**9 + 7

        # Precompute all distinct positive differences between pairs of elements.
        # These differences are the potential values for the power of subsequences.
        # Note: Subsequences with power 0 (due to duplicate elements or identical chosen elements)
        # contribute 0 to the sum and are implicitly handled by the formula focusing on d >= 1.
        diffs_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                diff = nums[j] - nums[i]
                # Only positive differences matter for the calculation N(>= d) where d >= 1.
                if diff > 0:
                    diffs_set.add(diff) 
        
        # If there are no positive differences (e.g., all elements are identical),
        # the power of any subsequence of length k >= 2 is 0. Thus, the total sum is 0.
        if not diffs_set:
             return 0

        # Sort the distinct positive differences. We will iterate through these values.
        sorted_diffs = sorted(list(diffs_set))
        
        total_sum_of_powers = 0
        prev_d = 0 # Tracks the previously considered difference value d_{i-1} in the formula

        # Helper function to compute N(>= d):
        # The number of subsequences of length k where the minimum difference is at least d.
        # This uses Dynamic Programming with optimizations.
        def count_subsequences_with_min_diff(d):
            # dp[i][j]: Stores the count of subsequences of length j ending with nums[i]
            #           that satisfy the minimum difference condition (>= d).
            #           Indices: i ranges from 0 to n-1, j ranges from 1 to k.
            dp = [[0] * (k + 1) for _ in range(n)]
            
            # prefix_sum[i][j]: Stores the cumulative sum of dp[p][j] for p from 0 to i, modulo MOD.
            # This optimization helps compute the transitions in O(log n) time instead of O(n).
            prefix_sum = [[0] * (k + 1) for _ in range(n)]

            # Base case: Subsequences of length 1.
            # Any single element nums[i] forms one subsequence of length 1.
            for i in range(n):
                dp[i][1] = 1
                if i == 0:
                    prefix_sum[i][1] = dp[i][1]
                else:
                    # Calculate prefix sum modulo MOD
                    prefix_sum[i][1] = (prefix_sum[i-1][1] + dp[i][1]) % MOD
            
            # Dynamic Programming: Fill table for lengths j = 2 to k.
            for j in range(2, k + 1):
                for i in range(n):
                    # To form a subsequence of length j ending at nums[i], we append nums[i]
                    # to a valid subsequence of length j-1 ending at nums[p] where p < i.
                    # The condition is nums[i] - nums[p] >= d, which implies nums[p] <= nums[i] - d.
                    
                    # Find the largest index p < i such that nums[p] <= nums[i] - d.
                    target = nums[i] - d
                    
                    # Use binary search (bisect_right) on the prefix nums[0...i-1].
                    # `bisect_right(a, x, hi=N)` searches within `a[0:N]`. Correct usage for range [0, i-1].
                    idx = bisect.bisect_right(nums, target, hi=i) 
                    
                    # `idx` is the count of elements in nums[0...i-1] that are <= target.
                    # These elements are at indices 0 to idx-1. The largest valid index p is `idx - 1`.
                    p_max = idx - 1
                        
                    if p_max >= 0:
                         # The number of valid subsequences ending at nums[i] is the sum
                         # of dp[p][j-1] for all valid p (0 <= p <= p_max).
                         # This sum is precalculated as prefix_sum[p_max][j-1].
                         dp[i][j] = prefix_sum[p_max][j-1]
                    # else: If no such p exists (p_max < 0), dp[i][j] remains 0.

                    # Update prefix sum for the current state (i, j).
                    if i == 0:
                        prefix_sum[i][j] = dp[i][j] # Will be 0 for i=0 as p_max is always < 0
                    else:
                        prefix_sum[i][j] = (prefix_sum[i-1][j] + dp[i][j]) % MOD

            # The total count N(>= d) is the sum of dp[i][k] for all ending indices i (0..n-1).
            # This is efficiently obtained from the last value of the prefix sum array for column k.
            return prefix_sum[n-1][k]

        # Calculate the total sum of powers using the derived formula:
        # Sum = sum_{idx=1..m} (d_idx - d_{idx-1}) * N(>= d_idx) % MOD
        # where d_idx are the distinct positive differences sorted, and d_0 = 0.
        for d in sorted_diffs:
            # Compute N(>= d), the count of subsequences with minimum difference >= d.
            count = count_subsequences_with_min_diff(d)
            
            # Calculate the contribution for the interval (prev_d, d].
            # The term (d - prev_d) * count represents the sum of powers contributed
            # by subsequences whose power p satisfies prev_d < p <= d.
            # Note: (d - prev_d) is guaranteed positive since sorted_diffs contains unique positive values.
            # The count is non-negative. The intermediate product could be large, but Python handles large integers.
            term = (d - prev_d) * count 
            total_sum_of_powers = (total_sum_of_powers + term) % MOD # Apply modulo after addition
            prev_d = d # Update prev_d for the next interval

        # The final result must be modulo MOD.
        # Since all terms added are non-negative, the sum is non-negative.
        # The final % MOD ensures the result is in [0, MOD-1].
        return total_sum_of_powers