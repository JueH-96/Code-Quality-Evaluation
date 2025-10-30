class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Build prefix sums for each character
        prefix = [[0] * (n + 1) for _ in range(26)]
        for i in range(n):
            for c in range(26):
                prefix[c][i+1] = prefix[c][i]
            char_index = ord(s[i]) - ord('a')
            prefix[char_index][i+1] += 1
        
        # Initialize DP array
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 characters need 0 partitions
        
        for i in range(1, n + 1):
            for j in range(i):
                # Check if substring s[j..i-1] is balanced
                counts = []
                for c in range(26):
                    cnt = prefix[c][i] - prefix[c][j]
                    if cnt > 0:
                        counts.append(cnt)
                # Determine if all counts are the same
                balanced = True
                if counts:
                    first = counts[0]
                    for num in counts[1:]:
                        if num != first:
                            balanced = False
                            break
                if balanced and dp[j] + 1 < dp[i]:
                    dp[i] = dp[j] + 1
        
        return dp[n]