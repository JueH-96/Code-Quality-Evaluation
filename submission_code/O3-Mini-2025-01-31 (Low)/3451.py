class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        n = len(word)
        while i < n:
            c = word[i]
            # Count how many times c occurs starting at i, maximum 9
            count = 0
            while i < n and word[i] == c and count < 9:
                count += 1
                i += 1
            comp.append(str(count))
            comp.append(c)
        return "".join(comp)