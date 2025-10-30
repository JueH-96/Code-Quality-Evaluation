class Solution:
    def possibleStringCount(self, word: str) -> int:
        if len(word) == 0:
            return 1
        
        # Count consecutive character groups
        count = 1  # Start with the original string
        i = 0
        
        while i < len(word):
            # Find the length of current group of same characters
            j = i
            while j < len(word) and word[j] == word[i]:
                j += 1
            
            group_length = j - i
            
            # If group has more than 1 character, we can reduce it
            if group_length > 1:
                # We can reduce by 1 to (group_length - 1) characters
                count += group_length - 1
            
            i = j
        
        return count