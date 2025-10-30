class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        # Helper function to compute the cyclic distance between two letters
        def dist(c1, c2):
            diff = abs(ord(c1) - ord(c2))
            return min(diff, 26 - diff)
        
        n = len(s)
        result = []
        
        for ch in s:
            # Try to transform current character 'ch' into the smallest possible letter
            # that does not exceed the remaining distance budget 'k'.
            for c in range(ord('a'), ord('z') + 1):
                candidate = chr(c)
                cost = dist(ch, candidate)
                if cost <= k:
                    # Use this candidate and reduce the budget
                    result.append(candidate)
                    k -= cost
                    break
        
        return "".join(result)