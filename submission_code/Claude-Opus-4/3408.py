class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lowercase = set()
        uppercase = set()
        
        for char in word:
            if char.islower():
                lowercase.add(char)
            else:
                uppercase.add(char.lower())
        
        return len(lowercase & uppercase)