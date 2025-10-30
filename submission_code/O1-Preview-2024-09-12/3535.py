class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10 ** 9 + 7
        n = len(nums)
        max_num = max(nums)

        dp = [{} for _ in range(n)]  # dp[i][a1] = number of ways ending with arr1[i]=a1

        # Initialize dp[0]
        for a1 in range(nums[0] + 1):
            dp[0][a1] = 1

        for i in range(1, n):
            dp_prev = dp[i - 1]
            dp_cur = dp[i]

            nums_diff = nums[i] - nums[i - 1]

            for prev_a1 in dp_prev:
                lb = max(prev_a1, prev_a1 + max(nums_diff, 0))
                if lb > nums[i]:
                    continue  # No valid a1 can be chosen
                ub = nums[i]
                for a1 in range(lb, ub + 1):
                    dp_cur[a1] = (dp_cur.get(a1, 0) + dp_prev[prev_a1]) % MOD

        total = sum(dp[n - 1].values()) % MOD
        return total