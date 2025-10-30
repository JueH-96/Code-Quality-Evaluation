class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        
        count = 0
        for char_code in range(ord('a'), ord('z') + 1):
            lower = chr(char_code)
            upper = chr(char_code - 32)
            if lower in word and upper in word:
                count += 1
        return count