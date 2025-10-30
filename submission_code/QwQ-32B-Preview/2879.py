class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        INF = float('inf')
        
        # Precompute possible d for each length
        possible_d = [[] for _ in range(n + 1)]
        for length in range(2, n + 1):
            for d in range(1, length):
                if length % d == 0:
                    possible_d[length].append(d)
        
        # Helper function to compute changes needed to make a sequence a palindrome
        def palindrome_changes(seq):
            changes = 0
            for i in range(len(seq) // 2):
                if seq[i] != seq[-(i + 1)]:
                    changes += 1
            return changes
        
        # Precompute cost(l, r)
        costs = [[INF for _ in range(n)] for _ in range(n)]
        for l in range(n):
            for r in range(l, n):
                length = r - l + 1
                if length == 1:
                    costs[l][r] = INF  # Cannot be semi-palindrome
                else:
                    min_changes = INF
                    for d in possible_d[length]:
                        total_changes = 0
                        for mod in range(d):
                            # Collect characters where index % d == mod
                            seq = [s[i] for i in range(l, r + 1, d)]
                            total_changes += palindrome_changes(seq)
                        if total_changes < min_changes:
                            min_changes = total_changes
                    costs[l][r] = min_changes
        
        # DP initialization
        dp = [[INF for _ in range(k + 1)] for _ in range(n + 1)]
        dp[0][0] = 0
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                for m in range(j - 1, i):
                    if m >= 0:
                        dp[i][j] = min(dp[i][j], dp[m][j - 1] + costs[m][i - 1])
        
        return dp[n][k]