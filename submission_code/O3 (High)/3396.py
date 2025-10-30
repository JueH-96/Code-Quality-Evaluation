class Solution:
    def isValid(self, word: str) -> bool:
        # Rule 1: length must be at least 3
        if len(word) < 3:
            return False
        
        vowels = set('aeiouAEIOU')
        has_vowel = False
        has_consonant = False
        
        for ch in word:
            if ch.isdigit():
                # digits are always allowed, nothing else to do
                continue
            elif ch.isalpha():
                # it's a letter → check if vowel/consonant
                if ch in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
            else:
                # any other character (@, #, $, or anything else) is invalid
                return False
        
        # Word is valid only if it has at least one vowel and one consonant
        return has_vowel and has_consonant