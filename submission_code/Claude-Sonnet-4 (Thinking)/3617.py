class Solution:
    def possibleStringCount(self, word: str) -> int:
        count = 1  # Count the possibility of no mistake
        
        # Count consecutive identical pairs
        for i in range(1, len(word)):
            if word[i] == word[i-1]:
                count += 1
                
        return count