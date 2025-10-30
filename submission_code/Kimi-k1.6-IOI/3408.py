class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lowercase = [False] * 26
        uppercase = [False] * 26
        for c in word:
            if c.islower():
                idx = ord(c) - ord('a')
                lowercase[idx] = True
            else:
                idx = ord(c) - ord('A')
                uppercase[idx] = True
        count = 0
        for i in range(26):
            if lowercase[i] and uppercase[i]:
                count += 1
        return count