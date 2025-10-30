class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter

        def is_valid_prefix(substr, word2):
            count_substr = Counter(substr)
            count_word2 = Counter(word2)
            for char in word2:
                if count_substr[char] < count_word2[char]:
                    return False
            return True

        count = 0
        n = len(word1)
        m = len(word2)

        for i in range(n):
            for j in range(i + m, n + 1):
                if is_valid_prefix(word1[i:j], word2):
                    count += 1

        return count