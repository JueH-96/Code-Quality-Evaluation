class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        count = 0

        word2_counts = {}
        for char in word2:
            word2_counts[char] = word2_counts.get(char, 0) + 1

        for i in range(n1):
            for j in range(i, n1):
                substring = word1[i:j+1]
                if len(substring) >= n2:
                    sub_counts = {}
                    for char in substring:
                        sub_counts[char] = sub_counts.get(char, 0) + 1

                    is_possible = True
                    for char, freq in word2_counts.items():
                        if sub_counts.get(char, 0) < freq:
                            is_possible = False
                            break
                    if is_possible:
                        count += 1
        return count