class Solution:
    def possibleStringCount(self, word: str) -> int:
        count = 1  # Start with 1 for the original string itself
        n = len(word)
        
        # Iterate through the string to find consecutive characters
        i = 0
        while i < n:
            j = i
            # Count the number of consecutive characters
            while j < n and word[j] == word[i]:
                j += 1
            
            # If there are more than 1 of the same character, we can form
            # additional strings by reducing the count of that character
            if j - i > 1:
                count += 1
            
            # Move to the next different character
            i = j
        
        return count