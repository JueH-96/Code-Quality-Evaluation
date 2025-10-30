class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        dp = [[0] * (1 << n) for _ in range(n)]
        graph = [[nums[i] % nums[j] == 0 for j in range(n)] for i in range(n)]

        def dfs(node, mask):
            if mask == (1 << n) - 1:
                return 1
            if dp[node][mask] != 0:
                return dp[node][mask]
            for i in range(n):
                if (mask & (1 << i)) == 0 and graph[node][i]:
                    dp[node][mask] += dfs(i, mask | (1 << i))
                    dp[node][mask] %= mod
            return dp[node][mask]

        return sum(dfs(i, 1 << i) for i in range(n)) % mod