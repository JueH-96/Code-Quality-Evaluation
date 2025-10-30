class Solution:
    def isValid(self, word: str) -> bool:
        # Check if length is at least 3
        if len(word) < 3:
            return False
        
        # Check if all characters are alphanumeric
        if not word.isalnum():
            return False
        
        # Check for at least one vowel and one consonant
        vowels = "aeiouAEIOU"
        has_vowel = any(char in vowels for char in word)
        has_consonant = any(char.isalpha() and char not in vowels for char in word)
        
        return has_vowel and has_consonant