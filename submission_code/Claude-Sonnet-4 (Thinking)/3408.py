class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        char_set = set(word)
        special_count = 0
        
        for char in char_set:
            if char.islower() and char.upper() in char_set:
                special_count += 1
        
        return special_count