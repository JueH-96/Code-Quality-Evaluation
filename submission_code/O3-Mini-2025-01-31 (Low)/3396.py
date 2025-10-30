class Solution:
    def isValid(self, word: str) -> bool:
        # Check minimum length requirement
        if len(word) < 3:
            return False
        
        # Define vowels (both lowercase and uppercase)
        vowels = set("aeiouAEIOU")
        
        # Flags for vowel and consonant existence
        has_vowel = False
        has_consonant = False
        
        # Check each character
        for ch in word:
            # Check if ch is alphanumeric (only digits and letters allowed)
            if not ch.isalnum():
                return False
            
            # If ch is a letter, then check whether it is vowel or consonant
            if ch.isalpha():
                if ch in vowels:
                    has_vowel = True
                else:
                    # Not in vowels and is alpha, then it's a consonant.
                    has_consonant = True
        
        # Must include at least one vowel and one consonant
        return has_vowel and has_consonant