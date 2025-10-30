class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        # Precompute all binary representations of powers of 5 up to length n
        p5bin = set()
        x = 1
        while True:
            b = bin(x)[2:]
            if len(b) > n:
                break
            p5bin.add(b)
            x *= 5
        
        INF = float('inf')
        # dp[i] = minimum number of beautiful substrings for s[:i]
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            # try every possible cut ending at i
            for j in range(i):
                # substring s[j:i]
                if s[j] == '0':
                    continue  # no leading zeros allowed
                sub = s[j:i]
                if sub in p5bin:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != INF else -1