class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        if n == 0:
            return 0
        
        original = [ord(c) - ord('a') for c in word]
        INF = float('inf')
        
        dp = [[INF] * 26 for _ in range(n)]
        
        # Initialize the first row
        first_char = original[0]
        for c in range(26):
            dp[0][c] = 0 if c == first_char else 1
        
        # Fill the DP table
        for i in range(1, n):
            current_original = original[i]
            for c in range(26):
                cost = 0 if c == current_original else 1
                min_prev = INF
                for prev_c in range(26):
                    if abs(c - prev_c) > 1:
                        if dp[i-1][prev_c] < min_prev:
                            min_prev = dp[i-1][prev_c]
                if min_prev != INF:
                    dp[i][c] = min_prev + cost
        
        return min(dp[-1])