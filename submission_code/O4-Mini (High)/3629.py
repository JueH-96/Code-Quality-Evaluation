class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        # dp[i] will hold the length of the string obtained by transforming
        # a single character with index i ('a'->0, 'b'->1, ..., 'z'->25) after k steps.
        # We start with k = 0: every character is length 1.
        dp = [1] * 26

        # Perform t transformations on the dp-array.
        # Transition:
        #  - for i in [0..24], f[k][i] = f[k-1][i+1]
        #  - for i = 25 ('z'), f[k][25] = f[k-1][0] + f[k-1][1]
        for _ in range(t):
            new_last = (dp[0] + dp[1]) % MOD
            # shift left by one: dp[i] = old dp[i+1] for i < 25
            dp = dp[1:]
            # append the 'z' case
            dp.append(new_last)

        # Sum up the contributions of each character in s
        ans = 0
        base = ord('a')
        for ch in s:
            ans = (ans + dp[ord(ch) - base]) % MOD

        return ans