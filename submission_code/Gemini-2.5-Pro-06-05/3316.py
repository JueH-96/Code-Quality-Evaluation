from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of powers of all subsequences of length k.

        The method rephrases the sum. Instead of summing powers for each subsequence,
        it sums, for each possible power value d, the number of subsequences with power >= d.
        Sum = Σ_{d>0} P(d), where P(d) is the count of subsequences with power >= d.

        P(d) is a step function changing only at d = |nums[i] - nums[j]|.
        We can calculate the sum by finding all unique differences, sorting them
        (d_1, d_2, ...), and computing Sum = Σ (d_i - d_{i-1}) * P(d_i).

        P(d) is calculated with dynamic programming. Let dp[i][j] be the number of
        valid subsequences of length j ending at nums[i].
        dp[i][j] = Σ dp[p][j-1] for p < i where nums[i] - nums[p] >= d.
        This is optimized with a two-pointer approach for a total complexity of O(n^2 * k * n).
        """
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()

        def count_subsequences_with_power_ge(d: int) -> int:
            # dp[i][j]: # of subsequences of length j, ending at nums[i], with power >= d
            dp = [[0] * (k + 1) for _ in range(n)]

            for i in range(n):
                dp[i][1] = 1

            for j in range(2, k + 1):
                prefix_sum = 0
                p_idx = 0
                for i in range(n):
                    while p_idx < i and nums[p_idx] + d <= nums[i]:
                        prefix_sum = (prefix_sum + dp[p_idx][j - 1]) % MOD
                        p_idx += 1
                    dp[i][j] = prefix_sum
            
            total_count = 0
            for i in range(n):
                total_count = (total_count + dp[i][k]) % MOD
            
            return total_count

        diffs_set = set()
        for i in range(n):
            for j in range(i):
                diffs_set.add(nums[i] - nums[j])
        
        sorted_unique_diffs = sorted(list(diffs_set))

        ans = 0
        prev_d = 0
        
        for d in sorted_unique_diffs:
            if d == 0:
                continue
            
            count = count_subsequences_with_power_ge(d)
            
            term = (d - prev_d) * count
            ans = (ans + term) % MOD
            
            prev_d = d
            
        return ans