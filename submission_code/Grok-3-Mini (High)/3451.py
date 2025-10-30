class Solution:
    def compressedString(self, word: str) -> str:
        res = []
        i = 0
        while i < len(word):
            c = word[i]
            prefix_len = 0
            while prefix_len < 9 and i + prefix_len < len(word) and word[i + prefix_len] == c:
                prefix_len += 1
            res.append(str(prefix_len) + c)
            i += prefix_len
        return ''.join(res)