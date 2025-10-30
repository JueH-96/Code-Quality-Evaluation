class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        chars = [ord(c) - ord('a') for c in word]
        dp = [[float('inf')] * 26 for _ in range(n)]
        
        # Initialize the first character
        first_char = chars[0]
        for c in range(26):
            dp[0][c] = 0 if c == first_char else 1
        
        for i in range(1, n):
            current_char = chars[i]
            for c in range(26):
                # Calculate the cost to change current character to c
                cost = 0 if c == current_char else 1
                min_prev = float('inf')
                # Find the minimum dp value from previous characters that are not almost equal to c
                for c_prev in range(26):
                    if abs(c_prev - c) > 1:
                        if dp[i-1][c_prev] < min_prev:
                            min_prev = dp[i-1][c_prev]
                if min_prev != float('inf'):
                    dp[i][c] = min_prev + cost
        
        return min(dp[-1])