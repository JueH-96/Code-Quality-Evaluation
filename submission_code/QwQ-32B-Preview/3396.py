class Solution:
    def isValid(self, word: str) -> bool:
        vowels = set("aeiouAEIOU")
        
        if len(word) < 3:
            return False
        
        has_vowel = False
        has_consonant = False
        
        for char in word:
            if char.isdigit() or char.isalpha():
                if char.isalpha():
                    if char in vowels:
                        has_vowel = True
                    else:
                        has_consonant = True
            else:
                # Invalid character found
                return False
        
        return has_vowel and has_consonant