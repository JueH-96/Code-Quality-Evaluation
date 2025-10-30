class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        ans = 0
        i = 0
        while i < n - 1:
            if abs(ord(word[i]) - ord(word[i + 1])) <= 1:
                ans += 1
                i += 2
            else:
                i += 1
        return ans