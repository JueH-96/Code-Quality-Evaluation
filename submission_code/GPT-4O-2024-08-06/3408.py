class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        # Create sets for lowercase and uppercase letters
        lowercase_set = set()
        uppercase_set = set()
        
        # Populate the sets with characters from the word
        for char in word:
            if char.islower():
                lowercase_set.add(char)
            elif char.isupper():
                uppercase_set.add(char.lower())
        
        # Find the intersection of both sets to get special characters
        special_chars = lowercase_set.intersection(uppercase_set)
        
        # Return the number of special characters
        return len(special_chars)