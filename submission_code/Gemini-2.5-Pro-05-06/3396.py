class Solution:
    def isValid(self, word: str) -> bool:
        # Rule 1: Minimum 3 characters
        if len(word) < 3:
            return False

        # Define the set of vowel characters for easy lookup.
        # Both lowercase and uppercase vowels are included.
        vowels = "aeiouAEIOU"
        
        # Flags to track if the word contains at least one vowel and one consonant.
        has_vowel = False
        has_consonant = False

        # Iterate through each character in the word.
        for char in word:
            # Rule 2: Contains only digits (0-9) and English letters.
            # The input `word` might contain '@', '#', '$' as per constraints.
            # `char.isalnum()` checks if `char` is an alphabet character or a digit.
            # If it's not, then the word contains a forbidden character.
            if not char.isalnum():
                return False  # Invalid character found

            # If the character is an English letter, determine if it's a vowel or consonant.
            if char.isalpha():
                if char in vowels:
                    has_vowel = True
                else:
                    # If it's an English letter and not a vowel, it's a consonant.
                    has_consonant = True
            # If `char.isdigit()` is true, it's a digit. This is allowed by Rule 2
            # and doesn't affect vowel/consonant counts.
            
        # Rule 3: Must include at least one vowel.
        if not has_vowel:
            return False

        # Rule 4: Must include at least one consonant.
        if not has_consonant:
            return False

        # If all conditions are met, the word is valid.
        return True