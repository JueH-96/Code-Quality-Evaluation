class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_complete(substring, k):
            from collections import Counter
            char_count = Counter(substring)
            if any(count != k for count in char_count.values()):
                return False
            for i in range(1, len(substring)):
                if abs(ord(substring[i]) - ord(substring[i - 1])) > 2:
                    return False
            return True

        n = len(word)
        count = 0

        for i in range(n):
            for j in range(i + k, min(i + 26 * k, n) + 1):
                if (j - i) % k == 0 and is_complete(word[i:j], k):
                    count += 1

        return count