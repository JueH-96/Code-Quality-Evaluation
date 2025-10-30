class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def distance(c1, c2):
            return min((ord(c1) - ord(c2)) % 26, (ord(c2) - ord(c1)) % 26)

        n = len(s)
        result = [''] * n
        remaining_k = k

        for i in range(n):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                dist = distance(s[i], c)
                if dist <= remaining_k:
                    result[i] = c
                    remaining_k -= dist
                    break

        return ''.join(result)