class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lowercase_set = set()
        uppercase_set = set()
        
        for char in word:
            if char.islower():
                lowercase_set.add(char)
            else:
                uppercase_set.add(char.lower())
        
        special = lowercase_set.intersection(uppercase_set)
        return len(special)