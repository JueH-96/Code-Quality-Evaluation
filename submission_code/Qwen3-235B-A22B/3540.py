class Solution:
    def stringHash(self, s: str, k: int) -> str:
        res = []
        n = len(s)
        for i in range(0, n, k):
            substring = s[i:i+k]
            total = 0
            for c in substring:
                total += ord(c) - ord('a')
            mod = total % 26
            res.append(chr(mod + ord('a')))
        return ''.join(res)