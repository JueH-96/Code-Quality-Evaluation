class Solution:
    def possibleStringCount(self, word: str) -> int:
        # Initialize the count of possible original strings
        possible_count = 1
        
        # Iterate through the word to find consecutive characters
        i = 0
        while i < len(word):
            # Count the number of consecutive same characters
            count = 1
            while i + 1 < len(word) and word[i] == word[i + 1]:
                count += 1
                i += 1
            
            # If there are more than one consecutive characters, it means
            # Alice might have pressed the key for too long, so we have
            # additional possible original strings
            if count > 1:
                possible_count += count
            
            # Move to the next character
            i += 1
        
        return possible_count