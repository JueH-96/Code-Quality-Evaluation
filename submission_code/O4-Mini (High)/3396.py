class Solution:
    def isValid(self, word: str) -> bool:
        # Check minimum length
        if len(word) < 3:
            return False
        
        vowels = set("aeiouAEIOU")
        has_vowel = False
        has_consonant = False
        
        for ch in word:
            if ch.isalpha():
                if ch in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
            elif ch.isdigit():
                # digits are allowed
                continue
            else:
                # any other character is invalid
                return False
        
        # Must have at least one vowel and one consonant
        return has_vowel and has_consonant