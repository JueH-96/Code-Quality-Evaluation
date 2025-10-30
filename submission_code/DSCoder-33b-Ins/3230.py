class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        count = [0]*26
        for ch in word:
            count[ord(ch)-ord('a')] += 1
        operations = 0
        for i in range(26):
            if count[i] > 0:
                operations += count[i] - 1
        return operations