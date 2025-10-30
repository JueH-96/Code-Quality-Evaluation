class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        while word:
            char = word[0]
            count = 1
            while count < 10 and len(word) > 1 and word[1] == char:
                count += 1
                word = word[1:]
            comp += str(count) + char
            word = word[count:]
        return comp