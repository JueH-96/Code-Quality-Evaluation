class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        while word:
            first_char = word[0]
            count = 0
            for char in word:
                if char == first_char and count < 9:
                    count += 1
                else:
                    break
            comp += str(count) + first_char
            word = word[count:]
        return comp