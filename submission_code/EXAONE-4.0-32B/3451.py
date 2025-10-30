class Solution:
    def compressedString(self, word: str) -> str:
        i = 0
        n = len(word)
        res = []
        while i < n:
            c = word[i]
            count = 0
            j = i
            while j < n and word[j] == c and count < 9:
                count += 1
                j += 1
            res.append(str(count) + c)
            i = j
        return ''.join(res)