from itertools import permutations

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        # First, put the three strings in a list.
        strings = [a, b, c]
        
        # Remove any string that is a substring of another.
        filtered = []
        for s in strings:
            # If there exists a different string t such that s is in t, then skip s.
            if any(s != t and s in t for t in strings):
                continue
            filtered.append(s)
        
        # Remove duplicates if any (order does not matter here).
        filtered = list(set(filtered))
        
        # Helper function to merge two strings.
        # This function returns the shortest string that contains s and t as substrings
        # when s is placed before t.
        def merge(s: str, t: str) -> str:
            # If t is already a substring of s, then s is enough.
            if t in s:
                return s
            # Similarly, if s is already in t, then return t.
            if s in t:
                return t
            # Find the maximum overlap where the suffix of s is a prefix of t.
            max_overlap = 0
            # Check from the largest possible overlap down to 1.
            for i in range(min(len(s), len(t)), 0, -1):
                if s.endswith(t[:i]):
                    max_overlap = i
                    break
            return s + t[max_overlap:]
        
        # We'll combine the strings in every possible order (permutation)
        # and choose the best candidate: that is, the one with minimum length,
        # and if tied, lexicographically smallest.
        best = None
        # Iterate over all permutations of the filtered strings.
        for perm in permutations(filtered):
            candidate = perm[0]
            # Merge the remaining strings sequentially.
            for i in range(1, len(perm)):
                candidate = merge(candidate, perm[i])
            # Update best candidate if this one is shorter or lexicographically smaller in case of tie.
            if best is None or len(candidate) < len(best) or (len(candidate) == len(best) and candidate < best):
                best = candidate
        
        return best