class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        n = len(word)
        
        while i < n:
            c = word[i]
            count = 1
            i += 1
            
            # Count up to 9 occurrences of the same character
            while i < n and word[i] == c and count < 9:
                count += 1
                i += 1
            
            comp.append(str(count))
            comp.append(c)
        
        return "".join(comp)