class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        n = len(word)
        while i < n:
            current_char = word[i]
            count = 0
            j = i
            while j < n and word[j] == current_char and count < 9:
                count += 1
                j += 1
            comp.append(f"{count}{current_char}")
            i += count
        return ''.join(comp)