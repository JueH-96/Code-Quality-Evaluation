class Solution:
    def isValid(self, word: str) -> bool:
        
        # Rule 1: It must contain a minimum of 3 characters.
        if len(word) < 3:
            return False

        # Flags to track the presence of a vowel and a consonant.
        has_vowel = False
        has_consonant = False
        
        # A string containing all vowels for easy lookup.
        vowels = "aeiouAEIOU"

        # Iterate through each character in the word to check the other rules.
        for char in word:
            # Rule 2: It must contain only digits (0-9) and English letters.
            # The str.isalnum() method checks if a character is alphanumeric.
            if not char.isalnum():
                return False
            
            # Check if the character is a letter to classify it as a vowel or consonant.
            if char.isalpha():
                if char in vowels:
                    # Rule 3: It must include at least one vowel.
                    has_vowel = True
                else:
                    # Rule 4: It must include at least one consonant.
                    # A consonant is a letter that is not a vowel.
                    has_consonant = True
        
        # After checking all characters, the word is valid only if it has both a vowel and a consonant.
        return has_vowel and has_consonant