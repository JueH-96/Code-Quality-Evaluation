from itertools import permutations

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def combine(s, t):
            if t in s:
                return s
            max_overlap = 0
            max_i = min(len(s), len(t))
            for i in range(1, max_i + 1):
                if s[-i:] == t[:i]:
                    max_overlap = i
            return s + t[max_overlap:]
        
        strings = [a, b, c]
        best = None
        
        for perm in permutations(strings):
            combined = combine(perm[0], perm[1])
            combined = combine(combined, perm[2])
            if best is None:
                best = combined
            else:
                if len(combined) < len(best):
                    best = combined
                elif len(combined) == len(best) and combined < best:
                    best = combined
        
        return best