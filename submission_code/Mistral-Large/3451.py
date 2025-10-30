class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        n = len(word)

        while i < n:
            count = 0
            char = word[i]

            while i < n and word[i] == char and count < 9:
                count += 1
                i += 1

            comp += str(count) + char

        return comp