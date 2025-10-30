class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lowercase_letters = set()
        uppercase_letters = set()
        
        for char in word:
            if char.islower():
                lowercase_letters.add(char)
            elif char.isupper():
                uppercase_letters.add(char.lower())
        
        # Find the intersection of the two sets
        special_letters = lowercase_letters.intersection(uppercase_letters)
        
        return len(special_letters)