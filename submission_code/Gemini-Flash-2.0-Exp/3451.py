class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        while word:
            count = 1
            while count < len(word) and count < 9 and word[count] == word[0]:
                count += 1
            if count == len(word) and count > 9 and word[0] == word[count-1]:
                count = 9
            comp += str(count) + word[0]
            word = word[count:]
        return comp