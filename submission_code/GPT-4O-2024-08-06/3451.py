class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        while i < len(word):
            current_char = word[i]
            count = 0
            # Count the number of consecutive characters, up to 9
            while i < len(word) and word[i] == current_char and count < 9:
                count += 1
                i += 1
            # Append the count and the character to the compressed string
            comp += str(count) + current_char
        return comp