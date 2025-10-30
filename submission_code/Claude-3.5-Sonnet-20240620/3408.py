class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lowercase = set()
        uppercase = set()
        special = set()

        for char in word:
            if char.islower():
                lowercase.add(char)
                if char.upper() in uppercase:
                    special.add(char.lower())
            elif char.isupper():
                uppercase.add(char)
                if char.lower() in lowercase:
                    special.add(char.lower())

        return len(special)