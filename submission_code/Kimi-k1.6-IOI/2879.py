class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        # Precompute the cost for each substring s[i...j]
        cost = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                length = j - i + 1
                if length < 2:
                    continue
                min_cost = float('inf')
                # Check all possible divisors d of length where 1 <= d < length
                for d in range(1, length):
                    if length % d == 0:
                        group_size = length // d
                        total = 0
                        # Check each group formed by indices with the same modulo d
                        for group in range(d):
                            # Each group has 'group_size' elements
                            for t in range(group_size // 2):
                                left = i + group + t * d
                                right = i + group + (group_size - 1 - t) * d
                                if s[left] != s[right]:
                                    total += 1
                        if total < min_cost:
                            min_cost = total
                cost[i][j] = min_cost
        
        # Dynamic programming to find the minimum changes for k partitions
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                # m is the end index of the (j-1)th partition, so the jth partition is from m to i-1
                for m in range(j - 1, i):
                    if dp[m][j - 1] != float('inf') and cost[m][i - 1] != float('inf'):
                        dp[i][j] = min(dp[i][j], dp[m][j - 1] + cost[m][i - 1])
        
        return dp[n][k]