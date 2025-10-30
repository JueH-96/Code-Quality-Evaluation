class Solution:
    def isValid(self, word: str) -> bool:
        # Check if the length is at least 3
        if len(word) < 3:
            return False
        
        # Prepare tracking for vowels and consonants
        vowels = set("aeiouAEIOU")
        found_vowel = False
        found_consonant = False
        
        # Iterate over each character in the word
        for ch in word:
            # Check if character is alphanumeric
            if not (ch.isalpha() or ch.isdigit()):
                return False
            
            # If character is a letter, check if it's vowel or consonant
            if ch.isalpha():
                if ch in vowels:
                    found_vowel = True
                else:
                    found_consonant = True
        
        # Return true only if at least one vowel and one consonant were found
        return found_vowel and found_consonant