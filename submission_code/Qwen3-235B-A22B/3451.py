class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        n = len(word)
        while i < n:
            c = word[i]
            count = 0
            while i + count < n and word[i + count] == c and count < 9:
                count += 1
            comp.append(f"{count}{c}")
            i += count
        return ''.join(comp)