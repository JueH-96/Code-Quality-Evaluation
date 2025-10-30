class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        def is_semi_palindrome(substring):
            n = len(substring)
            for d in range(1, n):
                if n % d == 0:
                    for i in range(d):
                        if substring[i] != substring[n - i - 1]:
                            break
                    else:
                        return True
            return False

        def min_changes_to_semi_palindrome(substring):
            n = len(substring)
            min_changes = float('inf')
            for d in range(1, n):
                if n % d == 0:
                    changes = 0
                    for i in range(d):
                        char_count = {}
                        for j in range(i, n, d):
                            char_count[substring[j]] = char_count.get(substring[j], 0) + 1
                        changes += len(char_count) - max(char_count.values())
                    min_changes = min(min_changes, changes)
            return min_changes

        memo = {}

        def dp(i, k):
            if (i, k) in memo:
                return memo[(i, k)]
            if i == len(s):
                return 0 if k == 0 else float('inf')
            if k == 0:
                return float('inf')
            min_val = float('inf')
            for j in range(i + 1, len(s) + 1):
                substring = s[i:j]
                changes = min_changes_to_semi_palindrome(substring)
                min_val = min(min_val, changes + dp(j, k - 1))
            memo[(i, k)] = min_val
            return min_val

        return dp(0, k)