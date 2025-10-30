class Solution:
    def isValid(self, word: str) -> bool:
        # Check the minimal length requirement.
        if len(word) < 3:
            return False
        
        vowels = set("aeiouAEIOU")
        vowel_found = False
        consonant_found = False
        
        # Check each character
        for ch in word:
            # Must be either a digit or a letter.
            if not (ch.isdigit() or ch.isalpha()):
                return False
            # If it's a letter, check whether it's a vowel or a consonant.
            if ch.isalpha():
                if ch in vowels:
                    vowel_found = True
                else:
                    consonant_found = True
        
        # The word is valid only if it contains at least one vowel and at least one consonant.
        return vowel_found and consonant_found