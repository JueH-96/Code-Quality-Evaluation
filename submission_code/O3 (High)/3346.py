class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        # helper to compute cyclic distance between two lowercase letters
        def dist(c1: str, c2: str) -> int:
            d = abs(ord(c1) - ord(c2))
            return min(d, 26 - d)

        res = []
        for c in s:
            # try every letter from 'a' to 'z'
            for code in range(ord('a'), ord('z') + 1):
                ch = chr(code)
                cost = dist(c, ch)
                if cost <= k:                 # we can afford this change
                    res.append(ch)            # choose the lexicographically smallest feasible letter
                    k -= cost                 # pay the cost
                    break                     # move to next position
        return ''.join(res)