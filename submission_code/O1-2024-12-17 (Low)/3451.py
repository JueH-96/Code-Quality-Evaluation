class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        n = len(word)
        
        while i < n:
            c = word[i]
            count = 1
            # Count up to 9 occurrences or until a different character appears
            while count < 9 and i + count < n and word[i + count] == c:
                count += 1
            comp.append(str(count) + c)
            i += count
        
        return "".join(comp)