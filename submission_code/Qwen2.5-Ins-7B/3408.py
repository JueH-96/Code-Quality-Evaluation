class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        seen = set()
        for char in word:
            if char.islower():
                seen.add(char.upper())
            elif char.isupper():
                seen.add(char.lower())
        return sum(1 for char in word if char.lower() in seen or char.upper() in seen)