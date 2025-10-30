class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        cost = [[(0) for _ in range(n)] for _ in range(n)]

        def is_semi_palindrome(sub):
            length = len(sub)
            for d in range(1, length):
                if length % d == 0:
                    valid = True
                    for i in range(d):
                        l, r = i, length - d + i
                        while l < r:
                            if sub[l] != sub[r]:
                                valid = False
                                break
                            l += d
                            r -= d
                        if not valid:
                            break
                    if valid:
                        return True
            return False

        def calculate_changes(sub):
            length = len(sub)
            min_changes = float('inf')
            for d in range(1, length):
                if length % d == 0:
                    changes = 0
                    for i in range(d):
                        l, r = i, length - d + i
                        while l < r:
                            if sub[l] != sub[r]:
                                changes += 1
                            l += d
                            r -= d
                    min_changes = min(min_changes, changes)
            if is_semi_palindrome(sub):
                return 0
            
            if length > 0:
                return min_changes if min_changes != float('inf') else length
            else:
                return 0

        for i in range(n):
            for j in range(i, n):
                cost[i][j] = calculate_changes(s[i : j + 1])

        dp = [[(float('inf')) for _ in range(k + 1)] for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, min(i // 2, k) + 1):
                for x in range(2 * j - 1, i + 1):
                    dp[i][j] = min(dp[i][j], dp[x - 1][j - 1] + cost[x - 1][i - 1])

        return dp[n][k]