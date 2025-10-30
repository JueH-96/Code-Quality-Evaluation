class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all binary strings of powers of 5 up to length 15
        beautiful = set()
        x = 1
        while True:
            b = bin(x)[2:]
            if len(b) > len(s):
                break
            beautiful.add(b)
            x *= 5
        
        n = len(s)
        # dp[i] = min number of beautiful substrings for s[i:]
        # initialize with infinity
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[n] = 0  # zero substrings needed for empty suffix
        
        # Fill dp from right to left
        for i in range(n - 1, -1, -1):
            # skip if leading zero
            if s[i] == '0':
                continue
            # try all end positions j > i
            for j in range(i + 1, n + 1):
                sub = s[i:j]
                if sub in beautiful:
                    dp[i] = min(dp[i], 1 + dp[j])
        
        return dp[0] if dp[0] < INF else -1

# Example usage:
# sol = Solution()
# print(sol.minimumBeautifulSubstrings("1011"))  # Output: 2
# print(sol.minimumBeautifulSubstrings("111"))   # Output: 3
# print(sol.minimumBeautifulSubstrings("0"))     # Output: -1