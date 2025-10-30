class Solution:
    def isValid(self, word: str) -> bool:
        # Check if the word contains at least 3 characters
        if len(word) < 3:
            return False

        # Define vowels
        vowels = set('aeiouAEIOU')

        # Initialize flags for vowels and consonants
        has_vowel = False
        has_consonant = False

        # Iterate through each character in the word
        for char in word:
            if not char.isalnum():
                return False
            if char.isdigit():
                continue
            if char in vowels:
                has_vowel = True
            else:
                has_consonant = True

        # Return True if the word has at least one vowel and one consonant
        return has_vowel and has_consonant