class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_count = {}
        upper_count = {}
        for char in word:
            if 'a' <= char <= 'z':
                lower_count[char] = lower_count.get(char, 0) + 1
            else:
                upper_count[char.lower()] = upper_count.get(char.lower(), 0) + 1
        
        count = 0
        for char in lower_count:
            if char in upper_count:
                count +=1
        return count