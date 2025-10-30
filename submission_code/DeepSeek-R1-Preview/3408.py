class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        chars = set(word)
        count = 0
        for c in 'abcdefghijklmnopqrstuvwxyz':
            if c in chars and c.upper() in chars:
                count += 1
        return count