class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter

        def can_form(s, t):
            s_count = Counter(s)
            t_count = Counter(t)
            for char in t_count:
                if s_count[char] < t_count[char]:
                    return False
            return True

        n, m = len(word1), len(word2)
        count = 0

        for i in range(n):
            for j in range(i + 1, n + 1):
                if j - i >= m and can_form(word1[i:j], word2):
                    count += 1

        return count