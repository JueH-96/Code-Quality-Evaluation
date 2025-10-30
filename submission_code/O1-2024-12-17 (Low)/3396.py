class Solution:
    def isValid(self, word: str) -> bool:
        # Condition 1: must have at least 3 characters
        if len(word) < 3:
            return False
        
        vowels = set("aeiouAEIOU")
        has_vowel = False
        has_consonant = False
        
        # Condition 2: all characters must be digits or letters
        # While checking, also track if there's at least one vowel and one consonant
        for ch in word:
            if not (ch.isdigit() or ch.isalpha()):
                return False
            # Check vowels
            if ch in vowels:
                has_vowel = True
            # If character is a letter but not a vowel, it's a consonant
            elif ch.isalpha():
                has_consonant = True
        
        # Condition 3 & 4: must contain at least one vowel and at least one consonant
        return has_vowel and has_consonant