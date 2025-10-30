class Solution:
    def possibleStringCount(self, word: str) -> int:
        count = 1  # Start with the original string (no mistakes)
        i = 0
        
        while i < len(word):
            # Find the end of the current consecutive group
            j = i
            while j < len(word) and word[j] == word[i]:
                j += 1
            
            # For a group of size n, there are n-1 ways Alice could have held the key too long
            consecutive_count = j - i
            if consecutive_count > 1:
                count += consecutive_count - 1
            
            i = j
        
        return count