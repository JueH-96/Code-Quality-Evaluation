class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        # dp[c] = length of result for character chr(c + ord('a')) after current number of transforms
        dp = [1] * 26  # at t=0, each char stays length 1
        
        for _ in range(t):
            new_dp = [0] * 26
            # for 'a' to 'y', it becomes the next char
            for c in range(25):
                new_dp[c] = dp[c+1]
            # 'z' becomes "ab"
            new_dp[25] = (dp[0] + dp[1]) % MOD
            dp = new_dp
        
        result = 0
        for ch in s:
            result = (result + dp[ord(ch) - ord('a')]) % MOD
        return result