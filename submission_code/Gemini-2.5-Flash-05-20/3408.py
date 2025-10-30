class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        # Convert the word into a set of characters for efficient O(1) average time lookups.
        char_set = set(word)
        
        special_count = 0
        
        # Iterate through all possible lowercase English letters from 'a' to 'z'.
        # We can use ASCII values to easily get corresponding lowercase and uppercase chars.
        # ord('a') is 97, ord('A') is 65. The difference is 32.
        for i in range(26):
            # Calculate the current lowercase character (e.g., 'a', 'b', 'c', ...)
            lower_char = chr(ord('a') + i)
            
            # Calculate the corresponding uppercase character (e.g., 'A', 'B', 'C', ...)
            upper_char = chr(ord('A') + i)
            
            # Check if both the lowercase and uppercase versions of the current letter
            # are present in the set of characters from the input word.
            if lower_char in char_set and upper_char in char_set:
                special_count += 1
                
        return special_count