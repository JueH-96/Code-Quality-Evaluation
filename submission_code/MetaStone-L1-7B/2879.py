class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        min_changes = [[0 for _ in range(n)] for __ in range(n)]
        
        for m in range(n):
            for i in range(m + 1, n):
                substring = s[m:i]
                min_changes[m][i] = self.compute_min_changes(substring)
        
        INF = float('inf')
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for j in range(1, k + 1):
            for i in range(j, n + 1):
                if j == 1:
                    if i >= 1:
                        cost = min_changes[0][i]
                        dp[i][j] = dp[0][0] + cost
                else:
                    for m in range(j - 1, i):
                        if dp[m][j - 1] == INF:
                            continue
                        cost = min_changes[m][i]
                        if dp[i][j] > dp[m][j - 1] + cost:
                            dp[i][j] = dp[m][j - 1] + cost
        
        return dp[n][k]
    
    def compute_min_changes(self, substring):
        m = len(substring)
        if m == 0:
            return 0
        
        def get_divisors(x):
            divisors = set()
            for i in range(1, int(x ** 0.5) + 1):
                if x % i == 0:
                    if i < x:
                        divisors.add(i)
                    if (x // i) < x and (x // i) != i:
                        divisors.add(x // i)
            return list(divisors)
        
        divisors = get_divisors(m)
        min_cost = float('inf')
        
        for d in divisors:
            groups = [[] for _ in range(d)]
            for idx, c in enumerate(substring):
                groups[idx % d].append(c)
            
            total = 0
            for group in groups:
                n_group = len(group)
                changes = 0
                for i in range(n_group // 2):
                    if group[i] != group[n_group - 1 - i]:
                        changes += 1
                total += changes
            
            if total < min_cost:
                min_cost = total
        
        return min_cost