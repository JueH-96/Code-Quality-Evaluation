class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        from functools import lru_cache
        
        def is_semi_palindrome(sub):
            n = len(sub)
            if n == 1:
                return 0
            for d in range(1, n):
                if n % d == 0:
                    changes = 0
                    for start in range(d):
                        group = [sub[i] for i in range(start, n, d)]
                        most_common = max(set(group), key=group.count)
                        changes += sum(1 for x in group if x != most_common)
                    return changes
            return float('inf')
        
        @lru_cache(None)
        def dp(i, k):
            if k == 1:
                return is_semi_palindrome(s[i:])
            if i == len(s):
                return float('inf')
            min_changes = float('inf')
            for j in range(i + 1, len(s) + 1):
                current_change = is_semi_palindrome(s[i:j])
                min_changes = min(min_changes, current_change + dp(j, k - 1))
            return min_changes
        
        return dp(0, k)