class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        from functools import lru_cache

        def is_semi_palindrome(subs):
            n = len(subs)
            for d in range(1, n):
                if n % d == 0:
                    valid = True
                    for i in range(d):
                        temp = subs[i:n:d]
                        if temp != temp[::-1]:
                            valid = False
                            break
                    if valid:
                        return True
            return False

        def count_changes_to_semi_palindrome(subs):
            n = len(subs)
            min_changes = float('inf')
            for d in range(1, n):
                if n % d == 0:
                    changes = 0
                    for i in range(d):
                        temp = subs[i:n:d]
                        changes += sum(temp[j] != temp[-j-1] for j in range(len(temp) // 2))
                    min_changes = min(min_changes, changes)
            return min_changes

        @lru_cache(None)
        def dp(start, k):
            if k == 1:
                return count_changes_to_semi_palindrome(s[start:])
            min_changes = float('inf')
            for end in range(start + 1, len(s) - k + 2):
                changes = count_changes_to_semi_palindrome(s[start:end]) + dp(end, k - 1)
                min_changes = min(min_changes, changes)
            return min_changes

        return dp(0, k)