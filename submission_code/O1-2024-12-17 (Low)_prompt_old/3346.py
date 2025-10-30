class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        # Helper function to compute cyclic distance between two characters
        def dist(c1, c2):
            d = abs(ord(c1) - ord(c2))
            return min(d, 26 - d)
        
        n = len(s)
        result = []
        
        for i in range(n):
            # Try all possible characters from 'a' to 'z'
            for c in range(ord('a'), ord('z') + 1):
                ch = chr(c)
                cost = dist(s[i], ch)
                if cost <= k:
                    result.append(ch)
                    k -= cost  # Reduce the remaining budget
                    break
        
        return "".join(result)