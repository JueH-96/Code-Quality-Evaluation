class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        n = len(word)
        while i < n:
            current_char = word[i]
            j = i
            while j < n and word[j] == current_char:
                j += 1
            count = j - i
            k = min(9, count)
            comp.append(f"{k}{current_char}")
            i += k
        return ''.join(comp)