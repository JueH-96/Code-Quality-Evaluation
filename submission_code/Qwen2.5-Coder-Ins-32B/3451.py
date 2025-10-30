class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        while i < len(word):
            c = word[i]
            count = 0
            while i < len(word) and word[i] == c and count < 9:
                count += 1
                i += 1
            comp += str(count) + c
        return comp