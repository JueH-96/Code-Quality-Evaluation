class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        res = list(s)
        sum_so_far = 0
        for i in range(len(s)):
            original = s[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == original:
                    distance = 0
                else:
                    d = ord(c) - ord(original)
                    distance = min(abs(d), 26 - abs(d))
                if sum_so_far + distance <= k:
                    res[i] = c
                    sum_so_far += distance
                    break
        return ''.join(res)