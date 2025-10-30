class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        mod = 10**9 + 7
        fact = [1]*(n+1)
        inv = [1]*(n+1)
        invfact = [1]*(n+1)
        for i in range(1,n+1):
            fact[i] = fact[i-1]*i%mod
            inv[i] = inv[i-1]*(mod-mod//i)%mod
            invfact[i] = invfact[i-1]*inv[i]%mod
        dp = [[0]*(n+1) for _ in range(n+1)]
        dp[0][0] = 1
        for i in range(1,n+1):
            dp[i][0] = 1
            for j in range(1,i+1):
                dp[i][j] = (dp[i-1][j] + dp[i-1][j-1]*j)%mod
        f = [1]*(n+1)
        for i in range(1,n+1):
            f[i] = f[i-1]*i%mod
        ans = 1
        for i in range(len(requirements)):
            end_i, cnt_i = requirements[i]
            if i == 0:
                ans = ans * fact[n-1] * invfact[n-1-end_i] % mod
                ans = ans * dp[n-end_i-1][cnt_i] % mod
            else:
                end_prev, cnt_prev = requirements[i-1]
                ans = ans * fact[end_prev-end_i] * invfact[end_prev-end_i-cnt_i+cnt_prev] % mod
                ans = ans * dp[end_i-end_prev][cnt_i-cnt_prev] % mod
        return ans