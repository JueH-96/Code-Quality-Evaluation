class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute is_balanced[i][j] which is True if s[i..j] is balanced
        is_balanced = [[False] * n for _ in range(n)]
        for i in range(n):
            counts = [0] * 26
            for j in range(i, n):
                char = s[j]
                counts[ord(char) - ord('a')] += 1
                # Check if all non-zero counts are the same
                non_zero = [cnt for cnt in counts if cnt != 0]
                if non_zero:
                    first = non_zero[0]
                    all_same = all(x == first for x in non_zero)
                    is_balanced[i][j] = all_same
        
        # Dynamic programming to find the minimal number of partitions
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: empty string requires 0 partitions
        for i in range(1, n + 1):
            for j in range(i):
                if is_balanced[j][i - 1]:
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
        
        return dp[n] if dp[n] != float('inf') else 0