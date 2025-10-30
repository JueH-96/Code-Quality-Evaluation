class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10 ** 9 + 7
        max_t = t
        dp = [[0] * (t + 1) for _ in range(26)]
        # Base case: at t=0, each character contributes length 1
        for i in range(26):
            dp[i][0] = 1

        for time in range(1, t + 1):
            for c in range(26):
                if c != 25:  # Not 'z'
                    dp[c][time] = dp[c + 1][time - 1]
                else:  # 'z' becomes 'a' + 'b'
                    dp[c][time] = (dp[0][time - 1] + dp[1][time - 1]) % MOD

        ans = 0
        for char in s:
            index = ord(char) - ord('a')
            ans = (ans + dp[index][t]) % MOD
        return ans