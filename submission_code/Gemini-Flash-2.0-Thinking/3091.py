class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1

        distinct_nums = sorted(counts.keys())
        n_distinct = len(distinct_nums)

        dp = [0] * (r + 1)
        dp[0] = 1

        for num in distinct_nums:
            freq = counts[num]
            for _ in range(freq):
                for s in range(r, num - 1, -1):
                    dp[s] = (dp[s] + dp[s - num]) % MOD

        ans = 0
        for s in range(l, r + 1):
            ans = (ans + dp[s]) % MOD

        return ans