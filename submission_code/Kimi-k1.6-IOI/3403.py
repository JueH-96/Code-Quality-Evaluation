class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # Create a prefix sum array where prefix[i][c] is the count of character c in s[0..i-1]
        prefix = [[0] * 26 for _ in range(n + 1)]
        for i in range(1, n + 1):
            for c in range(26):
                prefix[i][c] = prefix[i-1][c]
            char_index = ord(s[i-1]) - ord('a')
            prefix[i][char_index] += 1
        
        # Initialize DP array where dp[i] is the minimum substrings for the first i characters
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 characters need 0 substrings
        
        for i in range(1, n + 1):
            for j in range(i):
                # Check if substring s[j..i-1] is balanced
                counts = [prefix[i][c] - prefix[j][c] for c in range(26)]
                non_zero = [cnt for cnt in counts if cnt != 0]
                balanced = True
                if non_zero:
                    first = non_zero[0]
                    for cnt in non_zero[1:]:
                        if cnt != first:
                            balanced = False
                            break
                if balanced:
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
        
        return dp[n]