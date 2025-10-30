class Solution:
    def isValid(self, word: str) -> bool:
        # Check minimum length
        if len(word) < 3:
            return False
        
        # Define vowels
        vowels = set('aeiouAEIOU')
        
        # Track if we found vowels and consonants
        has_vowel = False
        has_consonant = False
        
        # Check each character
        for char in word:
            # Check if character is valid (digit or letter)
            if not (char.isdigit() or char.isalpha()):
                return False
            
            # If it's a letter, check if vowel or consonant
            if char.isalpha():
                if char in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
        
        # Word is valid if it has both vowel and consonant
        return has_vowel and has_consonant