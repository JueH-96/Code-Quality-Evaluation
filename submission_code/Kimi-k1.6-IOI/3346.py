class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        res = list(s)
        remaining = k
        for i in range(len(res)):
            if remaining <= 0:
                break
            original = res[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                diff = abs(ord(original) - ord(c))
                cost = min(diff, 26 - diff)
                if cost <= remaining:
                    res[i] = c
                    remaining -= cost
                    break
        return ''.join(res)