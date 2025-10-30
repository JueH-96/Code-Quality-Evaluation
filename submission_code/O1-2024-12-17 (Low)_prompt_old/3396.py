class Solution:
    def isValid(self, word: str) -> bool:
        # Condition 1: Check minimum length
        if len(word) < 3:
            return False
        
        vowels = set("aeiouAEIOU")
        vowel_found = False
        consonant_found = False
        
        for ch in word:
            # Condition 2: Must be digit or letter
            if not ch.isalnum():
                return False
            
            # Track vowels and consonants
            if ch.isalpha():
                if ch in vowels:
                    vowel_found = True
                else:
                    consonant_found = True
        
        # Condition 3: At least one vowel
        # Condition 4: At least one consonant
        return vowel_found and consonant_found