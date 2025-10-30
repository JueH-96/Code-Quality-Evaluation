class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        while i < len(word):
            j = i
            while j < len(word) and word[j] == word[i]:
                j += 1
            count = j - i
            comp += str(min(count, 9)) + word[i]
            i = j
        return comp