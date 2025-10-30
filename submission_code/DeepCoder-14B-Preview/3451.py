class Solution:
    def compressedString(self, word: str) -> str:
        comp = ''
        i = 0
        while i < len(word):
            current_char = word[i]
            j = i
            while j < i + 9 and j < len(word) and word[j] == current_char:
                j += 1
            count = j - i
            comp += str(count) + current_char
            i += count
        return comp