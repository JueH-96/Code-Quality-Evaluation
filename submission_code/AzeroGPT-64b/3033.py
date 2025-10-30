class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        if n == 1:
            return 0 if s1 == s2 else -1

        dp = [-1 for _ in range(n)]

        def dfs(i, cur_cost=-1):
            if i >= n:
                return cur_cost if cur_cost >= 0 else float("inf")
            if cur_cost >= 0 and dp[i] >= 0 and cur_cost >= dp[i]:
                return dp[i]
            
            if s1[i] == s2[i]:
                return dfs(i + 1, cur_cost)
            
            cost_single_flip = dfs(i + 1, cur_cost + 1)
            cost_double_flip = dfs(i + 2, cur_cost + x) if i + 1 < n else float("inf")
            
            if cur_cost >= 0:
                dp[i] = min(cost_single_flip, cost_double_flip)
                return dp[i]
            return min(cost_single_flip, cost_double_flip)

        res = dfs(0)
        if res == float("inf"):
            return -1
        return res