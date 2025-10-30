class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        from functools import lru_cache

        def is_semi_palindrome(sub):
            n = len(sub)
            for d in range(1, n):
                if n % d == 0:
                    is_palindrome = True
                    for i in range(d):
                        for j in range(i, n, d):
                            if sub[j] != sub[i]:
                                is_palindrome = False
                                break
                        if not is_palindrome:
                            break
                    if is_palindrome:
                        return True
            return False

        def min_changes_to_semi_palindrome(sub):
            n = len(sub)
            min_changes = float('inf')
            for d in range(1, n):
                if n % d == 0:
                    changes = 0
                    for i in range(d):
                        freq = {}
                        for j in range(i, n, d):
                            if sub[j] in freq:
                                freq[sub[j]] += 1
                            else:
                                freq[sub[j]] = 1
                        max_freq = max(freq.values())
                        changes += (n // d) - max_freq
                    min_changes = min(min_changes, changes)
            return min_changes

        @lru_cache(None)
        def dp(i, k):
            if k == 1:
                return min_changes_to_semi_palindrome(s[i:])
            min_changes = float('inf')
            for j in range(i + 1, len(s) - k + 2):
                min_changes = min(min_changes, min_changes_to_semi_palindrome(s[i:j]) + dp(j, k - 1))
            return min_changes

        return dp(0, k)