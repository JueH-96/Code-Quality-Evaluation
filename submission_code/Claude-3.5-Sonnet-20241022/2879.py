class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        def get_changes(substr):
            m = len(substr)
            if m == 1:
                return float('inf')
                
            min_changes = float('inf')
            # Try all possible d values
            for d in range(1, m):
                if m % d != 0:
                    continue
                    
                changes = 0
                # Group characters with same modulo d
                groups = [[] for _ in range(d)]
                for i in range(m):
                    groups[i % d].append(substr[i])
                
                # Check if each group forms palindrome
                for group in groups:
                    l = len(group)
                    for i in range(l // 2):
                        if group[i] != group[l-1-i]:
                            changes += 1
                            
                min_changes = min(min_changes, changes)
            
            return min_changes if min_changes != float('inf') else float('inf')
        
        # dp[i][j] represents min changes needed for first i chars split into j parts
        dp = [[float('inf')] * (k+1) for _ in range(n+1)]
        dp[0][0] = 0
        
        # For each length
        for i in range(1, n+1):
            # For each number of splits
            for j in range(1, min(i//2 + 1, k+1)):
                # Try different previous positions
                for prev in range(j-1, i):
                    substr = s[prev:i]
                    changes = get_changes(substr)
                    if changes != float('inf'):
                        dp[i][j] = min(dp[i][j], dp[prev][j-1] + changes)
        
        return dp[n][k]