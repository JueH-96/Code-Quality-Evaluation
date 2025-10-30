class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        # dp[i][c]: minimum changes up to position i if we set word[i] to character c (0 for 'a', ..., 25 for 'z')
        # Initialize dp with infinity
        INF = 10**9
        dp = [[INF] * 26 for _ in range(n)]
        
        # Base case for i = 0
        orig0 = ord(word[0]) - ord('a')
        for c in range(26):
            dp[0][c] = 0 if c == orig0 else 1
        
        # Fill dp table
        for i in range(1, n):
            orig = ord(word[i]) - ord('a')
            for c in range(26):
                cost = 0 if c == orig else 1
                # Try all previous letters p that are not almost-equal to c
                best_prev = INF
                for p in range(26):
                    if abs(p - c) > 1:
                        best_prev = min(best_prev, dp[i-1][p])
                dp[i][c] = best_prev + cost
        
        # The answer is the minimum cost at the last position over all possible letters
        return min(dp[n-1])