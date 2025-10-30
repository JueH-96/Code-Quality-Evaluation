class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        mod = 10**9 + 7
        fact = [0]*(n+1)
        inv = [0]*(n+1)
        fact[0] = fact[1] = inv[0] = inv[1] = 1
        for i in range(2, n+1):
            fact[i] = (fact[i-1]*i)%mod
            inv[i] = (mod - mod//i)*inv[mod%i]%mod
        for i in range(2, n+1):
            inv[i] = (inv[i-1]*inv[i])%mod
        requirements.sort()
        dp = [0]*n
        dp[0] = 1
        for i in range(1, n):
            dp[i] = dp[i-1]*fact[i]%mod
        for r in requirements:
            end, cnt = r
            for i in range(end+1):
                dp[i] = (dp[i]*(fact[i]-i+cnt)%mod*inv[i-cnt]%mod)%mod
        return dp[-1]