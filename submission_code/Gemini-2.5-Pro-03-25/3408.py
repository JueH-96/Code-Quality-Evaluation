import string # Import the string module to easily access lowercase letters

class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        """
        Counts the number of letters that appear in both lowercase and uppercase in the word.

        Args:
            word: The input string consisting of lowercase and uppercase English letters.

        Returns:
            The number of special letters.
        """
        
        # Create a set of all unique characters present in the word.
        # Sets provide efficient O(1) average time complexity for membership checking ('in').
        present_chars = set(word)
        
        special_count = 0
        
        # Iterate through all lowercase English letters ('a' through 'z').
        # string.ascii_lowercase provides the string "abcdefghijklmnopqrstuvwxyz".
        for lower_char in string.ascii_lowercase:
            # Get the corresponding uppercase letter.
            upper_char = lower_char.upper()
            
            # Check if both the lowercase version and the uppercase version
            # of the current letter exist in the set of characters from the word.
            if lower_char in present_chars and upper_char in present_chars:
                # If both are present, this letter is "special". Increment the count.
                special_count += 1
                
        # Return the final count of special letters.
        return special_count