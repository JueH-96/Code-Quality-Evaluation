class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        while i < len(word):
            j = i + 1
            while j < len(word) and word[j] == word[i] and j - i < 9:
                j += 1
            count = j - i
            comp += str(count) + word[i]
            i = j
        return comp