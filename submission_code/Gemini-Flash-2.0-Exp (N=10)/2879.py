class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        def is_semi_palindrome(sub):
            len_sub = len(sub)
            if len_sub <= 1:
                return False
            for d in range(1, len_sub):
                if len_sub % d == 0:
                    is_semi = True
                    for start in range(d):
                        temp = ""
                        for i in range(start, len_sub, d):
                            temp += sub[i]
                        
                        l, r = 0, len(temp) - 1
                        while l < r:
                            if temp[l] != temp[r]:
                                is_semi = False
                                break
                            l += 1
                            r -= 1
                        if not is_semi:
                            break
                    if is_semi:
                        return True
            return False

        def changes_to_semi_palindrome(sub):
            len_sub = len(sub)
            if len_sub <= 1:
                return float('inf')
            min_changes = float('inf')
            for d in range(1, len_sub):
                if len_sub % d == 0:
                    changes = 0
                    for start in range(d):
                        temp = ""
                        for i in range(start, len_sub, d):
                            temp += sub[i]
                        
                        l, r = 0, len(temp) - 1
                        while l < r:
                            if temp[l] != temp[r]:
                                changes += 1
                            l += 1
                            r -= 1
                    min_changes = min(min_changes, changes)
            return min_changes

        
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for p in range(i):
                    cost = changes_to_semi_palindrome(s[p:i])
                    if cost != float('inf'):
                        dp[i][j] = min(dp[i][j], dp[p][j-1] + cost)
        
        return dp[n][k]