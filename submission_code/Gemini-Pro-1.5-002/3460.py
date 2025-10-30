class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        reqs = sorted(requirements)
        
        dp = [[0] * 401 for _ in range(n)]
        
        for i in range(n):
            dp[i][0] = 1
            
        for i in range(1, n):
            prefix_sum = [0] * 401
            for j in range(401):
                prefix_sum[j] = (prefix_sum[j-1] if j > 0 else 0) + dp[i-1][j]
                prefix_sum[j] %= MOD
            
            for j in range(401):
                dp[i][j] = prefix_sum[j]
                if j > i:
                    dp[i][j] -= prefix_sum[j - i - 1]
                    dp[i][j] %= MOD
                    
        ans = 0
        prev_inv = 0
        
        for req in reqs:
            end, cnt = req
            
            if cnt < prev_inv:
                return 0
            
            ans += dp[end][cnt - prev_inv]
            ans %= MOD
            
            prev_inv = cnt
            
        return ans