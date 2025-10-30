class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        count=0
        for i in range(97,97+26):
            if chr(i) in word and chr(i-32) in word:
                count+=1
        return count