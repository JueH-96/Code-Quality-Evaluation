class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lowercase_chars = set()
        uppercase_chars = set()
        
        for char in word:
            if char.islower():
                lowercase_chars.add(char)
            else:
                uppercase_chars.add(char.lower())
        
        return len(lowercase_chars & uppercase_chars)