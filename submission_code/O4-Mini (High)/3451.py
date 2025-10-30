class Solution:
    def compressedString(self, word: str) -> str:
        res = []
        i, n = 0, len(word)
        while i < n:
            # take up to 9 of the same character
            j = i + 1
            while j < n and word[j] == word[i] and j - i < 9:
                j += 1
            count = j - i
            res.append(str(count))
            res.append(word[i])
            i = j
        return "".join(res)