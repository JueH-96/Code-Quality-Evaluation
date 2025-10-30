class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        while i < len(word):
            # Count occurrences of the current character
            count = 0
            char = word[i]
            while i < len(word) and word[i] == char and count < 9:
                count += 1
                i += 1
            # Append the count and the character to the compressed string
            comp += str(count) + char
        return comp