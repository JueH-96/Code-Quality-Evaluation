class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)
        prefix_count = [0] * n
        suffix_count = [0] * n
        count = [0] * 26

        # Count the number of occurrences of each character in word2
        for c in word2:
            count[ord(c) - ord('a')] += 1

        # Count the number of valid prefixes
        for i in range(n):
            count[ord(word1[i]) - ord('a')] += 1
            if i >= m and count[ord(word1[i - m]) - ord('a')] >= 1:
                count[ord(word1[i - m]) - ord('a')] -= 1
            if all(v == 0 for v in count):
                prefix_count[i] = 1

        # Count the number of valid suffixes
        for i in range(n - 1, -1, -1):
            count[ord(word1[i]) - ord('a')] += 1
            if i <= n - m and count[ord(word1[i + m]) - ord('a')] >= 1:
                count[ord(word1[i + m]) - ord('a')] -= 1
            if all(v == 0 for v in count):
                suffix_count[i] = 1

        # The total number of valid substrings is the sum of the prefix and suffix counts
        return sum(a and b for a, b in zip(prefix_count, suffix_count))