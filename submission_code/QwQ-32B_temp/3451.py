class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        n = len(word)
        while i < n:
            current_char = word[i]
            j = i
            # Find the maximum j such that the substring from i to j-1 is all current_char and length <=9
            while j < n and word[j] == current_char and (j - i) < 9:
                j += 1
            comp.append(f"{j - i}{current_char}")
            i = j
        return ''.join(comp)