class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        count = 0
        word1_len = len(word1)
        word2_len = len(word2)
        for i in range(word1_len - word2_len + 1):
            if word1[i:i+word2_len] == word2:
                count += 1
        return count