class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        
        # dp[first][last] = minimum length with first and last characters
        dp = [[float('inf')] * 26 for _ in range(26)]
        
        # Base case: start with the first word
        first = ord(words[0][0]) - ord('a')
        last = ord(words[0][-1]) - ord('a')
        dp[first][last] = len(words[0])
        
        # Process each subsequent word
        for i in range(1, n):
            new_dp = [[float('inf')] * 26 for _ in range(26)]
            w_first = ord(words[i][0]) - ord('a')
            w_last = ord(words[i][-1]) - ord('a')
            w_len = len(words[i])
            
            for f in range(26):
                for l in range(26):
                    if dp[f][l] == float('inf'):
                        continue
                    
                    # Option 1: append words[i] to the right
                    # If last char of current string matches first char of words[i], save 1 char
                    cost1 = dp[f][l] + w_len - (1 if l == w_first else 0)
                    new_dp[f][w_last] = min(new_dp[f][w_last], cost1)
                    
                    # Option 2: prepend words[i] to the left
                    # If last char of words[i] matches first char of current string, save 1 char
                    cost2 = dp[f][l] + w_len - (1 if w_last == f else 0)
                    new_dp[w_first][l] = min(new_dp[w_first][l], cost2)
            
            dp = new_dp
        
        # Find the minimum length among all possible first/last character combinations
        return min(min(row) for row in dp)