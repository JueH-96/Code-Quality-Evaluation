class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        while i < len(word):
            char = word[i]
            count = 0
            while i < len(word) and word[i] == char:
                i += 1
                count += 1
            comp += str(count) + char
        return comp