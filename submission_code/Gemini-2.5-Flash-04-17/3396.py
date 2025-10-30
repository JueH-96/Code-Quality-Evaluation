class Solution:
    def isValid(self, word: str) -> bool:
        # 1. Check minimum length
        if len(word) < 3:
            return False

        has_vowel = False
        has_consonant = False
        vowels = "aeiouAEIOU"

        # Iterate through characters
        for char in word:
            # 2. Check if character is alphanumeric (digit or letter)
            if not char.isalnum():
                return False

            # Check for vowels and consonants if the character is a letter
            if char.isalpha():
                if char in vowels:
                    has_vowel = True
                else: # It's a letter but not a vowel, so it's a consonant
                    has_consonant = True
            # Digits do not affect the vowel/consonant flags

        # 3 & 4. Check if at least one vowel and one consonant were found
        return has_vowel and has_consonant