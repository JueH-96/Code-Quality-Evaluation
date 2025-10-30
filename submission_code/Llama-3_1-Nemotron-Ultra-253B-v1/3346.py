class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        res = []
        remaining_k = k
        for c in s:
            for t in 'abcdefghijklmnopqrstuvwxyz':
                cost = min(abs(ord(t) - ord(c)), 26 - abs(ord(t) - ord(c)))
                if cost <= remaining_k:
                    res.append(t)
                    remaining_k -= cost
                    break
        return ''.join(res)