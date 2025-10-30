class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        import sys
        input_data = sys.stdin.read()  # Not strictly needed on some platforms, but good practice
        MOD = 10**9 + 7
        
        # In case the solution environment does not provide the inputs in the same way,
        # we assume nums and queries are already given as function arguments per the problem statement.
        # We'll implement the logic directly.

        n = len(nums)

        # dp[i] will hold the maximum sum of a non-adjacent subsequence considering elements up to index i
        dp = [0]*n
        if n >= 1:
            dp[0] = max(0, nums[0])  # can pick nums[0] if positive, or pick nothing
        if n >= 2:
            dp[1] = max(dp[0], nums[1], 0)  # pick best of dp[0], or nums[1], or nothing

        for i in range(2, n):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i], 0)

        ans = 0

        for pos, x in queries:
            # Update nums[pos]
            nums[pos] = x

            # Re-calculate dp from pos onward
            if pos == 0:
                dp[0] = max(0, nums[0])
                start_index = 1
            else:
                start_index = pos

            if start_index == 1 and n > 1:
                dp[1] = max(dp[0], nums[1], 0)

            for i in range(max(2, start_index), n):
                # dp[i] = max of dp[i-1], dp[i-2] + nums[i], 0
                dp[i] = max(dp[i-1], dp[i-2] + nums[i], 0)

            ans = (ans + dp[n-1]) % MOD

        return ans % MOD