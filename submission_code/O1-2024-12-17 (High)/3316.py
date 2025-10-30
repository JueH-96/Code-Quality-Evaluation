from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        """
        We want the sum of "powers" of all size-k subsequences of nums, where
        the "power" is the minimum absolute difference among the chosen elements.

        Approach:
        1. Sort the array.
        2. Gather all distinct pairwise differences d (including 0 if duplicates exist).
           Let these differences be d_1 < d_2 < ... < d_m. Also append a sentinel
           value d_{m+1} that is larger than any possible difference.
        3. Define count_subsets(d) to be the number of size-k subsequences whose
           consecutive elements (in sorted order) differ by at least d.
           - This is computed via DP in O(n * k) time:
             Let n = len(nums). We set dp[i][r] = number of ways to pick an r-subsequence
             ending exactly at index i (0-based) such that any two consecutive chosen
             elements differ by at least d.
             We also maintain prefix sums to make state transitions O(1).
        4. If we let C(d_i) = count_subsets(d_i), then the number of subsequences with
           min-difference exactly d_i is C(d_i) - C(d_{i+1}).
           (Because those with min-difference >= d_i but not >= d_{i+1} have min diff = d_i.)
        5. The answer is sum over i of [ d_i * (C(d_i) - C(d_{i+1})) ] modulo 1e9+7.

        This works because for a subsequence's minimum gap to be at least d,
        consecutive elements (in sorted order) must differ by >= d. Then to find
        how many have min-difference exactly d_i, we subtract those that have
        min-difference >= the next larger difference.

        Complexity:
            - We have at most n*(n-1)/2 distinct differences (at most 1225 for n=50).
            - For each difference, the DP takes O(n * k).
            - Overall O(n * k * (#differences)) is feasible for n up to 50.

        Examples match the given test cases.
        """

        mod = 10**9 + 7
        nums.sort()
        n = len(nums)

        # Collect all distinct nonnegative differences
        diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(nums[j] - nums[i])
        diffs = sorted(diffs)  # all distinct pairwise differences (>= 0)

        # Append a sentinel difference larger than any in diffs
        # Max possible difference is nums[-1] - nums[0], so add +1
        sentinel = (nums[-1] - nums[0]) + 1
        diffs.append(sentinel)

        # Precompute a helper to count how many k-subsequences
        # have consecutive differences >= d.
        def count_subsets(d: int) -> int:
            # p[i] = largest j < i such that nums[j] <= nums[i] - d
            # We'll fill this with a pointer approach
            p = [-1] * n
            t = 0
            for i2 in range(n):
                while t < i2 and nums[t] <= nums[i2] - d:
                    t += 1
                p[i2] = t - 1 if t > 0 else -1

            # dp[i][r] = number of ways to form an r-subsequence ending at i
            # prefix[r][i+1] = sum of dp[x][r] for x in [0..i]
            dp = [[0]*(k+1) for _ in range(n)]
            prefix = [[0]*(n+1) for _ in range(k+1)]

            # Base case: for r=1, there's exactly 1 way to pick a single element ending at i
            for i2 in range(n):
                dp[i2][1] = 1
            # Build prefix sums for r=1
            running = 0
            for i2 in range(n):
                running = (running + dp[i2][1]) % mod
                prefix[1][i2+1] = running

            # Fill dp for r=2..k
            for r in range(2, k+1):
                running_sum = 0
                idx_dp = 0
                for i2 in range(n):
                    if p[i2] != -1:
                        dp[i2][r] = prefix[r-1][p[i2]+1]  # sum of dp[x][r-1], x <= p[i2]
                    else:
                        dp[i2][r] = 0
                    dp[i2][r] %= mod

                    # Update prefix[r][i2+1]
                    running_sum = (running_sum + dp[i2][r]) % mod
                    prefix[r][i2+1] = running_sum

            # The total ways to pick k elements is sum of dp[i][k] for i=0..n-1
            return prefix[k][n] % mod

        ans = 0
        # Compute the sum of powers by differences
        for i in range(len(diffs) - 1):
            d_cur = diffs[i]
            d_next = diffs[i+1]
            c_cur = count_subsets(d_cur)
            c_next = count_subsets(d_next)
            # Number of subsequences whose min-diff is exactly d_cur
            # is c_cur - c_next
            count_exact = (c_cur - c_next) % mod
            # Add d_cur * count_exact
            ans = (ans + d_cur * count_exact) % mod

        return ans