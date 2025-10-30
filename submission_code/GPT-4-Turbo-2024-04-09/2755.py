class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        # dp[i] will store the minimum number of extra characters after processing up to the i-th character of s
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # No extra characters if no part of the string is considered
        
        # Convert dictionary to a set for faster lookup
        word_set = set(dictionary)
        
        for i in range(1, n + 1):
            # Check all possible substrings ending at position i-1
            for j in range(i):
                word = s[j:i]
                if word in word_set:
                    # If the word is in the dictionary, update dp[i]
                    dp[i] = min(dp[i], dp[j] + (i - j - len(word)))
                else:
                    # If not, consider all characters in the substring as extra
                    dp[i] = min(dp[i], dp[j] + (i - j))
        
        # The answer is the minimum extra characters considering the whole string
        return dp[n]