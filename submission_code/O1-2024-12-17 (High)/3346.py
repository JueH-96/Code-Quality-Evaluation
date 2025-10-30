class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        # Helper function to compute the ring distance
        # between two characters on the 'a' to 'z' cycle
        def ring_dist(a: str, b: str) -> int:
            d = abs(ord(a) - ord(b))
            return min(d, 26 - d)
        
        result = []
        
        for ch in s:
            # Try letters from 'a' to 'z' in ascending order
            # and pick the first we can afford with the remaining k
            for c in range(26):
                candidate = chr(ord('a') + c)
                cost = ring_dist(ch, candidate)
                if cost <= k:
                    result.append(candidate)
                    k -= cost
                    break
        
        return "".join(result)