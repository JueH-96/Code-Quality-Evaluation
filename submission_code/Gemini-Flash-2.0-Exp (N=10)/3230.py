class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        if not word:
            return 0
        
        count = 0
        i = 0
        while i < len(word) - 1:
            if abs(ord(word[i]) - ord(word[i+1])) <= 1:
                count += 1
                i += 2
            else:
                i += 1
        
        return count