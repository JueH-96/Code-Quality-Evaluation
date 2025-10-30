class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        if n == 1:
            return len(words[0])
        
        # Create a 3D DP array: dp[i][first][last] = min length after processing i words
        # with first and last characters being 'first' and 'last'
        INF = float('inf')
        dp = [[[INF for _ in range(26)] for _ in range(26)] for _ in range(n)]
        
        # Base case: first word
        first_char = ord(words[0][0]) - ord('a')
        last_char = ord(words[0][-1]) - ord('a')
        dp[0][first_char][last_char] = len(words[0])
        
        # Fill DP table
        for i in range(1, n):
            word = words[i]
            word_first = ord(word[0]) - ord('a')
            word_last = ord(word[-1]) - ord('a')
            
            for f in range(26):
                for l in range(26):
                    if dp[i-1][f][l] == INF:
                        continue
                    
                    # Option 1: Append word to current string
                    new_length = dp[i-1][f][l] + len(word)
                    if l == word_first:  # Optimization when last char equals first char
                        new_length -= 1
                    dp[i][f][word_last] = min(dp[i][f][word_last], new_length)
                    
                    # Option 2: Prepend word to current string
                    new_length = dp[i-1][f][l] + len(word)
                    if word_last == f:  # Optimization when last char equals first char
                        new_length -= 1
                    dp[i][word_first][l] = min(dp[i][word_first][l], new_length)
        
        # Find minimum length after processing all words
        min_length = INF
        for f in range(26):
            for l in range(26):
                min_length = min(min_length, dp[n-1][f][l])
        
        return min_length