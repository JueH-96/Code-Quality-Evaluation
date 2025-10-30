class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        n = len(word)
        while i < n:
            current_char = word[i]
            count = 0
            while i + count < n and word[i + count] == current_char and count < 9:
                count += 1
            comp.append(f"{count}{current_char}")
            i += count
        return ''.join(comp)