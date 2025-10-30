class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        while i < len(word):
            char = word[i]
            count = 0
            j = i
            while j < len(word) and word[j] == char and count < 9:
                count += 1
                j += 1
            comp += str(count) + char
            i = j
        return comp