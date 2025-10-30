class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        dp = [0] * (t + 1)
        dp[0] = n
        for i in range(1, t + 1):
            for char in s:
                if char == 'z':
                    dp[i] = (dp[i] + 2) % MOD
                else:
                    dp[i] = (dp[i] + 1) % MOD
            s = ""
            for char in s:
                if char == 'z':
                    s += "ab"
                else:
                    s += chr(ord(char) + 1)
            if i < t:
                s = "".join(['ab' if c == 'z' else chr(ord(c) + 1) for c in s])

        return dp[t]