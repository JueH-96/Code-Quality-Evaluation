class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        # We'll build the result character by character.
        # For each position, we'll try all possible letters from 'a' to 'z'
        # and pick the first letter c such that the cyclic distance
        # from s[i] to c is <= k. Then we'll reduce k accordingly.
        
        def cyclic_distance(a, b):
            # Computes the minimum cyclic distance between two characters.
            diff = abs(ord(a) - ord(b))
            return min(diff, 26 - diff)
        
        result = []
        for ch in s:
            # Try letters from 'a' to 'z' to get the lexicographically smallest.
            for candidate_ord in range(ord('a'), ord('z') + 1):
                candidate = chr(candidate_ord)
                cost = cyclic_distance(ch, candidate)
                if cost <= k:
                    result.append(candidate)
                    k -= cost
                    break
        return "".join(result)