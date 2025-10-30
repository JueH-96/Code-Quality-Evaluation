class Solution:
    def isValid(self, word: str) -> bool:
        # Check minimum length condition
        if len(word) < 3:
            return False
        
        vowels = set("aeiouAEIOU")
        seen_vowel = False
        seen_consonant = False
        
        for ch in word:
            # Only allow digits and English letters
            if not (ch.isdigit() or ch.isalpha()):
                return False
            # Check for vowels and consonants only if character is a letter
            if ch.isalpha():
                if ch in vowels:
                    seen_vowel = True
                else:
                    seen_consonant = True
        
        # The word is valid only if it includes at least one vowel and one consonant.
        return seen_vowel and seen_consonant