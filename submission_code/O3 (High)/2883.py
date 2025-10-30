class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)

        # Pre–compute all binary representations of powers of 5
        # whose length does not exceed the length of the input string.
        pow5_bin = set()
        k = 0
        while True:
            b = bin(5 ** k)[2:]          # binary form without the "0b" prefix
            if len(b) > n:               # no longer needed – already too long
                break
            pow5_bin.add(b)
            k += 1

        INF = 10 ** 9
        dp = [INF] * (n + 1)             # dp[i] = min #parts for s[i:]
        dp[n] = 0                        # empty suffix needs 0 parts

        # Iterate from right to left building the answer for every suffix
        for i in range(n - 1, -1, -1):
            if s[i] == '0':               # cannot start with a leading zero
                continue
            for j in range(i + 1, n + 1): # consider every possible end-position
                sub = s[i:j]
                if sub in pow5_bin:
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[0] if dp[0] != INF else -1