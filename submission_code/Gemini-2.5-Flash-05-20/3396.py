class Solution:
    def isValid(self, word: str) -> bool:
        # Condition 1: Minimum length check
        if len(word) < 3:
            return False

        # Define sets for efficient vowel checking
        vowels_lower = {'a', 'e', 'i', 'o', 'u'}
        vowels_all_cases = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}

        has_vowel = False
        has_consonant = False

        for char in word:
            # Condition 2: Only digits and English letters
            # char.isalnum() checks if char is alphanumeric (letter or digit)
            if not char.isalnum():
                return False

            # Condition 3: At least one vowel
            if char in vowels_all_cases:
                has_vowel = True
            
            # Condition 4: At least one consonant
            # A consonant is an English letter that is not a vowel.
            # char.isalpha() ensures it's a letter (not a digit)
            # char.lower() not in vowels_lower ensures it's not a vowel
            if char.isalpha() and char.lower() not in vowels_lower:
                has_consonant = True
        
        # After checking all characters, return true only if both vowel and consonant were found
        return has_vowel and has_consonant