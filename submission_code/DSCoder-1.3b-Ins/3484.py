class Solution:
    def getSmallestString(self, s: str) -> str:
        n = len(s)
        res = [0] * n
        res[n - 1] = 2 * (ord('z') - ord('a'))
        for i in range(n - 2, -1, -1):
            if s[i] == '1':
                res[i] = 2 * (ord('z') - ord('a')) - res[i + 1]
            else:
                res[i] = ord('z') - ord(s[i])
        return ''.join([chr(res[i] + ord('a')) for i in range(n)])