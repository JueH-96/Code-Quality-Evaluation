class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all binary representations of powers of 5 up to length of s
        valid = set()
        x = 1
        n = len(s)
        while True:
            b = bin(x)[2:]
            if len(b) > n:
                break
            valid.add(b)
            x *= 5
        
        # dp[i] = minimum number of beautiful substrings to cover s[:i], inf if impossible
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j in range(i):
                # skip if prefix s[:j] not coverable
                if dp[j] == INF:
                    continue
                # substring s[j:i] must not have leading zero
                if s[j] == '0':
                    continue
                sub = s[j:i]
                if sub in valid:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != INF else -1