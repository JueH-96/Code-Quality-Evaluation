class Solution:
    def possibleStringCount(self, word: str) -> int:
        # Base case: empty word (though per constraints word has length >= 1)
        if not word:
            return 0
        
        # Count the case of no overtyping
        result = 1
        
        # Each adjacent equal pair in the final word corresponds
        # to one extra possibility (an overtype in that run).
        for i in range(1, len(word)):
            if word[i] == word[i - 1]:
                result += 1
        
        return result