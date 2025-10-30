class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        # Helper to compute cyclic distance between two chars
        def dist(a: str, b: str) -> int:
            d = abs(ord(a) - ord(b))
            return min(d, 26 - d)

        res = []
        for ch in s:
            # Try the smallest possible replacement from 'a' to 'z'
            for c in map(chr, range(ord('a'), ord('z') + 1)):
                cost = dist(ch, c)
                # If we can afford this cost, choose c and deduct from k
                if cost <= k:
                    res.append(c)
                    k -= cost
                    break
        return "".join(res)