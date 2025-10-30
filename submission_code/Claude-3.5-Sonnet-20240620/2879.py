class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        def cost(i, j):
            substr = s[i:j+1]
            length = j - i + 1
            min_changes = float('inf')
            
            for d in range(1, length):
                if length % d == 0:
                    changes = 0
                    for start in range(d):
                        segment = substr[start::d]
                        changes += sum(a != b for a, b in zip(segment, segment[::-1])) // 2
                    min_changes = min(min_changes, changes)
            
            return min_changes if min_changes != float('inf') else 0
        
        @functools.lru_cache(None)
        def dp(i, k):
            if k == 1:
                return cost(i, n-1)
            if i >= n:
                return float('inf')
            
            res = float('inf')
            for j in range(i, n-k+1):
                res = min(res, cost(i, j) + dp(j+1, k-1))
            return res
        
        return dp(0, k)