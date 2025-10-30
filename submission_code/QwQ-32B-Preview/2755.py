class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        # Convert dictionary to a set for O(1) lookups
        word_set = set(dictionary)
        n = len(s)
        # Initialize dp array with infinity
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: no extra characters before start
        
        for i in range(1, n + 1):
            # Assume the i-th character is extra
            dp[i] = dp[i - 1] + 1
            # Check all possible words in the dictionary
            for word in word_set:
                word_len = len(word)
                if i >= word_len and s[i - word_len:i] == word:
                    dp[i] = min(dp[i], dp[i - word_len])
        
        return dp[n]