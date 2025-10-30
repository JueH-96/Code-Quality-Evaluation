class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        operations = 0
        i = 1
        while i < len(word):
            if abs(ord(word[i]) - ord(word[i - 1])) <= 1:
                operations += 1
                i += 2
            else:
                i += 1
        return operations