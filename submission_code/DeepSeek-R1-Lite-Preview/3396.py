class Solution:
    def isValid(self, word: str) -> bool:
        # Define vowels
        vowels = set('aeiouAEIOU')
        
        # Check for minimum length
        if len(word) < 3:
            return False
        
        # Check if all characters are alphanumeric (letters and digits)
        if not word.isalnum():
            return False
        
        # Flags to check for presence of vowel and consonant
        has_vowel = False
        has_consonant = False
        
        # Iterate through each character in the word
        for char in word:
            if char in vowels:
                has_vowel = True
            elif char.isalpha():
                has_consonant = True
        
        # Return True if both vowel and consonant are present
        return has_vowel and has_consonant