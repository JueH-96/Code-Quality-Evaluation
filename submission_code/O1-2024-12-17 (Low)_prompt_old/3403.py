class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # Precompute prefix sums for character frequencies: prefixFreq[i][c] 
        # is the frequency of character c in s[:i].
        # We'll map 'a' -> 0, 'b' -> 1, ..., 'z' -> 25.
        prefixFreq = [[0]*26 for _ in range(n+1)]
        
        for i in range(n):
            for c in range(26):
                prefixFreq[i+1][c] = prefixFreq[i][c]
            prefixFreq[i+1][ord(s[i]) - ord('a')] += 1
        
        # dp[i] will hold the minimum number of balanced substrings for s[:i]
        dp = [float('inf')] * (n+1)
        dp[0] = 0  # empty string can be considered as 0 substrings
        
        for i in range(1, n+1):
            for j in range(i):
                # length of substring = i - j
                # freq of each character in s[j:i]
                freq = [prefixFreq[i][c] - prefixFreq[j][c] for c in range(26)]
                
                distinct_chars = sum(1 for c in freq if c > 0)
                length = i - j
                # Check if s[j:i] can be balanced
                if distinct_chars > 0 and length % distinct_chars == 0:
                    needed = length // distinct_chars
                    if all(f == 0 or f == needed for f in freq):
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]