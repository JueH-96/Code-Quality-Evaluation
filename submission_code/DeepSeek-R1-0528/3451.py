class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        n = len(word)
        while i < n:
            c = word[i]
            count = 0
            j = i
            while j < n and word[j] == c and count < 9:
                count += 1
                j += 1
            comp.append(str(count))
            comp.append(c)
            i += count
        return "".join(comp)