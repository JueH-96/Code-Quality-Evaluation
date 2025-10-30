class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        # Initialize dp table
        dp = [[0] * 26 for _ in range(n)]
        
        # Set the dp values for the first character
        first_char_num = ord(word[0]) - ord('a')
        for val in range(26):
            dp[0][val] = 0 if val == first_char_num else 1
        
        # Fill the dp table for each subsequent character
        for i in range(1, n):
            curr_char_num = ord(word[i]) - ord('a')
            for curr_val in range(26):
                change_cost = 0 if curr_val == curr_char_num else 1
                # Find the minimum of dp[i-1][prev] where |prev - curr_val| >= 2
                min_prev = min(dp[i-1][p] for p in range(26) if abs(p - curr_val) >= 2)
                dp[i][curr_val] = change_cost + min_prev
        
        # The result is the minimum value in the last row of dp
        return min(dp[n-1])