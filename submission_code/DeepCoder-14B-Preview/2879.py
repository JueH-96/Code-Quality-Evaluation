class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        if n == 0 or k == 0:
            return 0
        
        # Precompute the minimal changes for all possible substrings
        min_changes = [[float('inf')] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                len_sub = j - i + 1
                if len_sub < 2:
                    continue  # Substring must be at least 2 characters
                # Find all proper divisors (d < len_sub and len_sub % d == 0)
                divisors = []
                for d in range(1, len_sub):
                    if len_sub % d == 0:
                        divisors.append(d)
                if not divisors:
                    continue  # No valid d found (shouldn't happen for len >=2)
                min_cost = float('inf')
                for d in divisors:
                    changes = 0
                    s_sub = s[i:j+1]
                    for mod in range(d):
                        group = []
                        for idx in range(len(s_sub)):
                            if idx % d == mod:
                                group.append(idx)
                        m = len(group)
                        for x in range(m // 2):
                            if s_sub[group[x]] != s_sub[group[m - 1 - x]]:
                                changes += 1
                    if changes < min_cost:
                        min_cost = changes
                if min_cost != float('inf'):
                    min_changes[i][j] = min_cost
        
        # Initialize DP table
        INF = float('inf')
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # Base case: 0 characters, 0 partitions
        
        for i in range(1, n + 1):
            for curr_k in range(1, k + 1):
                for j in range(i):
                    # The substring is s[j..i-1], length i-j
                    if (i - j) < 2:
                        continue  # Substring must be at least 2 characters
                    if min_changes[j][i-1] == float('inf'):
                        continue  # This substring can't be a semi-palindrome
                    if dp[j][curr_k - 1] + min_changes[j][i-1] < dp[i][curr_k]:
                        dp[i][curr_k] = dp[j][curr_k - 1] + min_changes[j][i-1]
        
        result = dp[n][k]
        return result if result != float('inf') else -1