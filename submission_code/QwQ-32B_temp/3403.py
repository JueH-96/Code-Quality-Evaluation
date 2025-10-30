class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # base case: 0 characters require 0 partitions
        
        for i in range(1, n + 1):
            current_counts = [0] * 26
            min_partitions = dp[i]  # Initialize with current value (infinity)
            # Iterate j from i-1 down to 0
            for j in range(i - 1, -1, -1):
                # Add the current character to the counts
                c = s[j]
                idx = ord(c) - ord('a')
                current_counts[idx] += 1
                
                # Check if the current substring (from j to i-1) is balanced
                first = -1
                balanced = True
                for cnt in current_counts:
                    if cnt == 0:
                        continue
                    if first == -1:
                        first = cnt
                    else:
                        if cnt != first:
                            balanced = False
                            break
                if balanced:
                    # Update the minimum partitions if this is better
                    if dp[j] + 1 < min_partitions:
                        min_partitions = dp[j] + 1
            # Update dp[i] with the found minimum
            dp[i] = min_partitions
        
        return dp[n]