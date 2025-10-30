class Solution:
    def isValid(self, word: str) -> bool:
        # Must be at least 3 characters long
        if len(word) < 3:
            return False

        vowels = set("aeiouAEIOU")
        has_vowel = False
        has_consonant = False

        for ch in word:
            # If it's a letter, check for vowel or consonant
            if ch.isalpha():
                if ch in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
            # If it's a digit, that's fine; just continue
            elif ch.isdigit():
                continue
            # Any other character is invalid
            else:
                return False

        # Must have at least one vowel and one consonant
        return has_vowel and has_consonant