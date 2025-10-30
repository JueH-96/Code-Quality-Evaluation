class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        while word:
            count = 0
            char = word[0]
            for i in range(len(word)):
                if word[i] == char and count < 9:
                    count += 1
                else:
                    break
            comp += str(count) + char
            word = word[count:]
        return comp