class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        # Helper function that merges two strings with maximal overlap.
        def merge(s: str, t: str) -> str:
            if t in s:
                return s
            # Find maximum overlap of s's suffix with t's prefix.
            max_overlap = 0
            for k in range(1, min(len(s), len(t)) + 1):
                if s[-k:] == t[:k]:
                    max_overlap = k
            return s + t[max_overlap:]
        
        # Remove any string that is a substring of another.
        def filterSubstrings(strings):
            filtered = []
            for s in strings:
                if any(s != t and s in t for t in strings):
                    continue
                filtered.append(s)
            return filtered
        
        from itertools import permutations
        
        strings = [a, b, c]
        strings = filterSubstrings(strings)
        
        best = None
        # Try all permutations of the strings.
        for perm in permutations(strings):
            merged = merge(merge(perm[0], perm[1]), perm[2])
            if best is None:
                best = merged
            else:
                # We want the string with smallest length, then lex smallest.
                if len(merged) < len(best) or (len(merged) == len(best) and merged < best):
                    best = merged
        return best