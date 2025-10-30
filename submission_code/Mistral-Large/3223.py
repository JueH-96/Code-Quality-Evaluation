class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_complete(substr, k):
            from collections import Counter
            count = Counter(substr)
            for char, freq in count.items():
                if freq != k:
                    return False
            return True

        def is_valid_adjacent(substr):
            for i in range(len(substr) - 1):
                if abs(ord(substr[i]) - ord(substr[i + 1])) > 2:
                    return False
            return True

        n = len(word)
        count = 0

        for i in range(n):
            for j in range(i + k, n + 1):
                substr = word[i:j]
                if len(substr) % k != 0:
                    continue
                if is_complete(substr, k) and is_valid_adjacent(substr):
                    count += 1

        return count