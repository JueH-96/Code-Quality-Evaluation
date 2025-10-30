class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_complete(s, k):
            from collections import Counter
            count = Counter(s)
            if all(v == k for v in count.values()):
                for i in range(len(s) - 1):
                    if abs(ord(s[i]) - ord(s[i + 1])) > 2:
                        return False
                return True
            return False

        n = len(word)
        count = 0
        for i in range(n):
            for j in range(i + 1, n + 1):
                if is_complete(word[i:j], k):
                    count += 1
        return count