class Solution:
    def compressedString(self, word: str) -> str:
        i = 0
        n = len(word)
        comp = []
        while i < n:
            current_char = word[i]
            count = 0
            # Count up to 9 consecutive occurrences of current_char
            while i < n and word[i] == current_char and count < 9:
                count += 1
                i += 1
            comp.append(str(count) + current_char)
        return "".join(comp)