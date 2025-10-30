class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        chars = set(word)
        count = 0
        
        for char in chars:
            if char.islower() and char.upper() in chars:
                count += 1
        
        return count