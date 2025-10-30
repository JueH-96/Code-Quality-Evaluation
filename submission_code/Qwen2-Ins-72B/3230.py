class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        ans = 0
        i = 0
        while i < len(word) - 1:
            diff = ord(word[i+1]) - ord(word[i])
            if abs(diff) <= 1:
                ans += 1
                if diff == 0:
                    i += 1
            i += 1
        return ans