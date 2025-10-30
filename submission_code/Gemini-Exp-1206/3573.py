class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        count = 0

        if n2 > n1:
            return 0

        freq2 = {}
        for char in word2:
            freq2[char] = freq2.get(char, 0) + 1

        for i in range(n1):
            for j in range(i, n1):
                substring = word1[i : j + 1]
                if len(substring) < n2:
                    continue

                freq1 = {}
                for char in substring:
                    freq1[char] = freq1.get(char, 0) + 1

                valid = True
                for char, freq in freq2.items():
                    if char not in freq1 or freq1[char] < freq:
                        valid = False
                        break

                if valid:
                    count += 1

        return count