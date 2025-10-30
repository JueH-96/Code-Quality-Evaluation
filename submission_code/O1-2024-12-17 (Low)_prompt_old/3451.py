class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        n = len(word)
        
        while i < n:
            c = word[i]
            count = 0
            while i < n and word[i] == c and count < 9:
                i += 1
                count += 1
            comp.append(str(count))
            comp.append(c)
        
        return "".join(comp)