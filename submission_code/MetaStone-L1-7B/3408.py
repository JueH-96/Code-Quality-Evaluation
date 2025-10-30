class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lowercase = set()
        uppercase = set()
        for c in word:
            if c.islower():
                lowercase.add(c)
            else:
                uppercase.add(c)
        count = 0
        for c in lowercase:
            if c.upper() in uppercase:
                count += 1
        return count