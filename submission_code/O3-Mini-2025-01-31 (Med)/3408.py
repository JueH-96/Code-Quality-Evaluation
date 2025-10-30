class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        # Create a set for encountered lowercase letters
        lower_set = set()
        # Create a set for encountered letters that appear in uppercase (converted to lowercase)
        upper_set = set()
        
        # Iterate over each character in the word
        for ch in word:
            if ch.islower():
                lower_set.add(ch)
            elif ch.isupper():
                upper_set.add(ch.lower())
        
        # The special letters are those that appear in both the lower_set and upper_set
        special_letters = lower_set.intersection(upper_set)
        
        return len(special_letters)