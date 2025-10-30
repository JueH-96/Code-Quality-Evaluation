class Solution:
    def possibleStringCount(self, word: str) -> int:
        # Initialize the count of possible original strings
        possible_count = 1
        
        # Initialize the previous character and its count
        prev_char = word[0]
        prev_count = 1
        
        # Iterate over the word starting from the second character
        for i in range(1, len(word)):
            if word[i] == prev_char:
                # If the current character is the same as the previous one, increase the count
                prev_count += 1
            else:
                # If the current character is different, calculate the possible original strings
                # for the previous character sequence
                possible_count *= prev_count + 1
                # Update the previous character and reset its count
                prev_char = word[i]
                prev_count = 1
        
        # For the last sequence of characters
        possible_count *= prev_count + 1
        
        return possible_count