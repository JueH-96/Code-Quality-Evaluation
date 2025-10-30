import math

class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # Build prefix sums for each character
        prefix = [[0] * (n + 1) for _ in range(26)]
        for i in range(1, n + 1):
            char_idx = ord(s[i - 1]) - ord('a')
            for c in range(26):
                prefix[c][i] = prefix[c][i - 1]
            prefix[char_idx][i] += 1
        
        # DP array
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # Fill DP table
        for i in range(1, n + 1):  # For each ending index i
            for j in range(i):  # For each starting index j of the last substring
                # Check if substring s[j..i-1] is balanced
                min_cnt = float('inf')
                max_cnt = 0
                for c in range(26):
                    cnt = prefix[c][i] - prefix[c][j]  # Count of char c in s[j..i-1]
                    if cnt > 0:
                        if cnt < min_cnt:
                            min_cnt = cnt
                        if cnt > max_cnt:
                            max_cnt = cnt
                if min_cnt == max_cnt:  # All characters have the same frequency
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]