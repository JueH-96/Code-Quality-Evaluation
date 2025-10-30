class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        # Precompute divisors for all possible lengths
        from collections import defaultdict
        divisors = defaultdict(list)
        for length in range(2, n+1):
            for d in range(1, length):
                if length % d == 0:
                    divisors[length].append(d)
        
        # Precompute cost for all substrings
        cost = [[0] * (n+1) for _ in range(n)]
        
        for l in range(n):
            for r in range(l+1, n+1):
                len_sub = r - l
                min_changes = float('inf')
                for d in divisors[len_sub]:
                    if d >= len_sub:
                        continue
                    total_changes_d = 0
                    for i in range(d):
                        group = []
                        for m in range(len_sub // d):
                            group.append(s[l + i + m * d])
                        # Compute changes to make group palindrome
                        changes = 0
                        m_len = len(group)
                        for p in range(m_len // 2):
                            if group[p] != group[m_len - 1 - p]:
                                changes +=1
                        total_changes_d += changes
                    min_changes = min(min_changes, total_changes_d)
                cost[l][r] = min_changes if min_changes != float('inf') else float('inf')
        
        # Initialize DP
        dp = [[float('inf')] * (n+1) for _ in range(k+1)]
        dp[0][0] = 0
        
        for j in range(1, k+1):
            for i in range(1, n+1):
                for m in range(j-1, i):
                    if dp[j-1][m] + cost[m][i] < dp[j][i]:
                        dp[j][i] = dp[j-1][m] + cost[m][i]
        return dp[k][n]