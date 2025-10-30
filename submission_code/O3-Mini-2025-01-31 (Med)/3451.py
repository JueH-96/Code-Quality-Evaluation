class Solution:
    def compressedString(self, word: str) -> str:
        i = 0
        n = len(word)
        comp = []
        while i < n:
            current_char = word[i]
            count = 1
            # count additional same characters but limit to 9 in total
            while i + count < n and word[i + count] == current_char and count < 9:
                count += 1
            comp.append(f"{count}{current_char}")
            i += count
        return "".join(comp)