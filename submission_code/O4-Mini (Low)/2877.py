from itertools import permutations

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s: str, t: str) -> str:
            # If one contains the other, the superstring is the longer one
            if s.find(t) != -1:
                return s
            if t.find(s) != -1:
                return t
            # Otherwise find maximum overlap of suffix of s with prefix of t
            max_ov = 0
            max_len = min(len(s), len(t))
            for k in range(max_len, 0, -1):
                if s.endswith(t[:k]):
                    max_ov = k
                    break
            return s + t[max_ov:]
        
        candidates = []
        for x, y, z in permutations([a, b, c], 3):
            xy = merge(x, y)
            xyz = merge(xy, z)
            candidates.append(xyz)
        
        # Pick the minimal length, then lexicographically smallest
        ans = min(candidates, key=lambda s: (len(s), s))
        return ans