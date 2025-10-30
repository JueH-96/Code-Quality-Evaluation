class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        n = len(word)
        i = 0
        while i < n:
            current_char = word[i]
            count = 0
            while i < n and word[i] == current_char and count < 9:
                count += 1
                i += 1
            comp.append(f"{count}{current_char}")
        return ''.join(comp)