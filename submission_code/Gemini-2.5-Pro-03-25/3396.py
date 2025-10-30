import string

class Solution:
    def isValid(self, word: str) -> bool:
        """
        Checks if a word is valid based on the given criteria.

        Args:
            word: The input string to validate.

        Returns:
            True if the word is valid, False otherwise.
        """

        # 1. Check minimum length
        if len(word) < 3:
            return False

        # Define vowels and consonants (or just vowels)
        vowels = "aeiouAEIOU"
        # Consonants are letters that are not vowels. Digits are neither.

        # Flags to track presence of required character types
        has_vowel = False
        has_consonant = False
        
        # 2. Check character types and presence of vowels/consonants
        for char in word:
            # Check if the character is allowed (digit or letter)
            if not ('a' <= char <= 'z' or 'A' <= char <= 'Z' or '0' <= char <= '9'):
            # Alternative using built-ins: if not char.isalnum():
                return False # Invalid character found

            # Check if it's a letter
            if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
            # Alternative: if char.isalpha():
                # Check if it's a vowel
                if char in vowels:
                    has_vowel = True
                else:
                    # It's a letter but not a vowel, so it's a consonant
                    has_consonant = True
            # Digits are allowed, but don't contribute to vowel/consonant count

        # 3. Check if both a vowel and a consonant were found
        return has_vowel and has_consonant