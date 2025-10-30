class Solution:
    def possibleStringCount(self, word: str) -> int:
        count = 1  # Case where no key is pressed too long
        i = 0
        
        while i < len(word):
            j = i
            # Find the length of the current group of consecutive characters
            while j < len(word) and word[j] == word[i]:
                j += 1
            
            length = j - i
            # If the group has more than 1 character, it could have been formed
            # by pressing a key too long
            if length >= 2:
                count += length - 1
            
            i = j
        
        return count