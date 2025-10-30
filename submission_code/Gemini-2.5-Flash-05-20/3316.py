import bisect

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        MOD = 10**9 + 7

        nums.sort()  # Sort nums to ensure elements are in non-decreasing order

        # Generate all unique absolute differences between elements.
        # These differences are the only values 'X' for which C(X) might change.
        distinct_diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                distinct_diffs.add(nums[j] - nums[i])
        
        # If there are duplicate numbers in nums, a difference of 0 is possible.
        # Ensure 0 is in distinct_diffs if it's not already generated (e.g., [2,2] -> 0).
        # Even if not explicitly added, starting prev_d at 0 in the summation loop
        # correctly handles the range from 0 to the first positive difference.
        
        sorted_distinct_diffs = sorted(list(distinct_diffs))

        total_sum = 0
        prev_d = 0  # Represents d_{r-1} in the formula, starting with d_0 = 0

        # Calculate the sum using the formula: sum_{i=1 to M} (d_i - d_{i-1}) * C(d_i)
        # Where C(d_i) is the count of subsequences with minimum difference >= d_i.
        for d_current in sorted_distinct_diffs:
            # count_ge_d_current is C(d_current) in the formula
            count_ge_d_current = self._count_valid_subsequences(nums, k, d_current)
            
            # Add (d_current - d_prev) * C(d_current) to the total sum
            total_sum = (total_sum + (d_current - prev_d) * count_ge_d_current) % MOD
            
            prev_d = d_current

        return total_sum

    # Helper function to count subsequences with minimum difference >= X
    # nums: sorted list of numbers
    # k: desired length of subsequences
    # X: minimum required difference between adjacent elements in the sorted subsequence
    def _count_valid_subsequences(self, nums: List[int], k: int, X: int) -> int:
        n = len(nums)
        MOD = 10**9 + 7

        # dp[i][j] = number of subsequences of length j ending with nums[i]
        # that satisfy the minimum difference X condition.
        dp = [[0] * (k + 1) for _ in range(n)]

        # prefix_sum_dp[j][i] = sum(dp[p][j] for p in range(i+1))
        # This table allows efficient summation of dp values from previous length.
        prefix_sum_dp = [[0] * (k + 1) for _ in range(n)]

        # Base case: subsequences of length 1
        # Any single element forms a subsequence of length 1.
        for i in range(n):
            dp[i][1] = 1
            # Update prefix sum for length 1
            prefix_sum_dp[1][i] = (prefix_sum_dp[1][i-1] + dp[i][1]) % MOD if i > 0 else dp[i][1]
        
        # Fill DP table for lengths from 2 up to k
        for j in range(2, k + 1):
            # 'p_ptr' tracks the rightmost index in nums that can precede nums[i]
            # while satisfying the minimum difference X.
            p_ptr = 0 
            for i in range(n):
                # Move p_ptr forward as long as nums[p_ptr] is a valid previous element
                # (i.e., nums[p_ptr] <= nums[i] - X) and p_ptr is before i.
                while p_ptr < i and nums[p_ptr] <= nums[i] - X:
                    p_ptr += 1
                
                # After the loop, p_ptr is the first index such that nums[p_ptr] > nums[i] - X
                # (or p_ptr == i, meaning no valid previous elements were found).
                # So, we need to sum dp[t][j-1] for t from 0 to p_ptr - 1.
                # This sum is available from prefix_sum_dp[j-1][p_ptr-1].
                
                if p_ptr - 1 >= 0:
                    dp[i][j] = prefix_sum_dp[j-1][p_ptr-1]
                else:
                    dp[i][j] = 0 # No element prior to nums[i] satisfies the condition

                # Update prefix sum for current length j
                prefix_sum_dp[j][i] = (prefix_sum_dp[j][i-1] + dp[i][j]) % MOD if i > 0 else dp[i][j]
        
        # The total count of valid subsequences of length k
        # is the sum of all dp[i][k] for i from 0 to n-1,
        # which is stored in prefix_sum_dp[k][n-1].
        return prefix_sum_dp[k][n-1]