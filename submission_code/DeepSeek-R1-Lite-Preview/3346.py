class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        
        # Helper function to compute cyclic distance
        def distance(c1, c2):
            d = abs(ord(c1) - ord(c2))
            return min(d, 26 - d)
        
        # Initialize DP table
        # dp[i][k'] will store the lex smallest string from i with remaining budget k'
        dp = [["" for _ in range(k+1)] for _ in range(n+1)]
        
        # Base case: empty string for the last position
        for kk in range(k+1):
            dp[n][kk] = ""
        
        # Fill DP table
        for i in range(n-1, -1, -1):
            for kk in range(k+1):
                best = None
                for c in range(26):
                    char = chr(ord('a') + c)
                    cost = distance(s[i], char)
                    if cost <= kk:
                        candidate = char + dp[i+1][kk - cost]
                        if best is None or candidate < best:
                            best = candidate
                dp[i][kk] = best if best is not None else ""
        
        return dp[0][k]