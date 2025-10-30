class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        # Convert the dictionary to a set for faster lookup
        word_set = set(dictionary)
        n = len(s)
        
        # dp[i] will store the minimum number of extra characters for the substring s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: no extra characters for an empty string
        
        # Iterate over each position in the string
        for i in range(1, n + 1):
            # Assume the current character is extra
            dp[i] = dp[i - 1] + 1
            
            # Check all possible substrings ending at position i
            for j in range(i):
                if s[j:i] in word_set:
                    dp[i] = min(dp[i], dp[j])
        
        # The answer is the minimum extra characters for the entire string
        return dp[n]