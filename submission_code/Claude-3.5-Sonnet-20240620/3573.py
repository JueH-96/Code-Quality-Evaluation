class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        def is_valid(freq1, freq2):
            return all(freq1[c] >= freq2[c] for c in freq2)

        n, m = len(word1), len(word2)
        if n < m:
            return 0

        freq2 = {}
        for c in word2:
            freq2[c] = freq2.get(c, 0) + 1

        freq1 = {}
        count = 0

        for i in range(n):
            freq1[word1[i]] = freq1.get(word1[i], 0) + 1

            if i >= m - 1:
                if is_valid(freq1, freq2):
                    count += i - m + 2

                freq1[word1[i - m + 1]] -= 1
                if freq1[word1[i - m + 1]] == 0:
                    del freq1[word1[i - m + 1]]

        return count