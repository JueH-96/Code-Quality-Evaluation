class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        # Helper to compute cyclic distance between two lowercase letters
        def dist(c1: str, c2: str) -> int:
            d = abs(ord(c1) - ord(c2))
            return min(d, 26 - d)
        
        res = []
        rem = k
        for ch in s:
            # Try to assign the smallest possible character at this position
            for i in range(26):
                candidate = chr(ord('a') + i)
                cost = dist(ch, candidate)
                if cost <= rem:
                    res.append(candidate)
                    rem -= cost
                    break
        return "".join(res)