class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        dp = [0] * 26
        for char in s:
            dp[ord(char) - ord('a')] += 1
        for _ in range(t):
            new_dp = [0] * 26
            for i in range(25):
                new_dp[i + 1] = (new_dp[i + 1] + dp[i]) % MOD
            new_dp[0] = (new_dp[0] + dp[25]) % MOD
            new_dp[1] = (new_dp[1] + dp[25]) % MOD
            dp = new_dp
        ans = 0
        for count in dp:
            ans = (ans + count) % MOD
        return ans