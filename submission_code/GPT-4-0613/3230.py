class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        count = 0
        for i in range(1, len(word)):
            if abs(ord(word[i]) - ord(word[i-1])) <= 1:
                count += 1
        return count