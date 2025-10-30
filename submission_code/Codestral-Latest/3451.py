class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        while word:
            # Find the length of the maximum prefix of the same character
            char = word[0]
            count = 0
            while count < 9 and count < len(word) and word[count] == char:
                count += 1
            # Append the count and the character to comp
            comp += str(count) + char
            # Remove the prefix from word
            word = word[count:]
        return comp