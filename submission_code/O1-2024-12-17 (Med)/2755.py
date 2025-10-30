class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        word_set = set(dictionary)
        
        # dp[i] will hold the minimum number of extra characters
        # for the substring s[i:]
        dp = [0] * (n + 1)

        # We'll fill dp from the end to the beginning
        for i in range(n - 1, -1, -1):
            # By default, skip this character (count it as extra) and recurse
            dp[i] = 1 + dp[i + 1]
            
            # Also check all dictionary words to see if we can match one at i
            for w in word_set:
                if s.startswith(w, i):
                    # If word w matches s[i:i+len(w)], skip those characters
                    dp[i] = min(dp[i], dp[i + len(w)])
        
        return dp[0]