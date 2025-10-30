class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        dp = [[float('inf')] * 26 for _ in range(n)]
        
        first_char = ord(word[0]) - ord('a')
        for c in range(26):
            if c == first_char:
                dp[0][c] = 0
            else:
                dp[0][c] = 1
        
        for i in range(1, n):
            current_char_index = ord(word[i]) - ord('a')
            for c in range(26):
                cost = 0 if c == current_char_index else 1
                best_prev = float('inf')
                for prev in range(26):
                    if abs(prev - c) > 1:
                        if dp[i-1][prev] < best_prev:
                            best_prev = dp[i-1][prev]
                dp[i][c] = best_prev + cost
        
        return min(dp[-1])