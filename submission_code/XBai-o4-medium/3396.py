class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        
        vowels = {'a', 'e', 'i', 'o', 'u'}
        vowel_count = 0
        consonant_count = 0
        
        for c in word:
            # Check if character is not alphanumeric
            if not (c.isalpha() or c.isdigit()):
                return False
            # Check if the character is a letter (not a digit)
            if c.isalpha():
                lower_c = c.lower()
                if lower_c in vowels:
                    vowel_count += 1
                else:
                    consonant_count += 1
        
        return vowel_count >= 1 and consonant_count >= 1