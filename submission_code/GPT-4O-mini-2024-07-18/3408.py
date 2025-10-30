class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        special_letters = set()
        lower_set = set()
        upper_set = set()
        
        for char in word:
            if char.islower():
                lower_set.add(char)
            elif char.isupper():
                upper_set.add(char.lower())
        
        # Find intersection of lower_set and upper_set
        special_letters = lower_set.intersection(upper_set)
        
        return len(special_letters)