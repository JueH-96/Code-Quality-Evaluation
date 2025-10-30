class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        # dp[i] will hold the minimum number of extra chars
        # for s[:i] (the substring from 0 to i-1).
        dp = [0] * (n + 1)
        
        # Initialize dp so that in the worst case we leave all characters as extra.
        # That is, dp[i] = i means all i chars are considered extra if no match is found.
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + 1  # By default, count the current character as extra
            for w in dictionary:
                w_len = len(w)
                # If the current substring matches a dictionary word, update dp
                if i >= w_len and s[i - w_len:i] == w:
                    dp[i] = min(dp[i], dp[i - w_len])
        
        return dp[n]