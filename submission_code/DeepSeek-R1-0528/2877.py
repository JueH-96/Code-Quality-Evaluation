class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s1, s2):
            if s2 in s1:
                return s1
            if s1 in s2:
                return s2
            k = min(len(s1), len(s2))
            while k > 0:
                if s1[-k:] == s2[:k]:
                    return s1 + s2[k:]
                k -= 1
            return s1 + s2
        
        permutations = [
            [a, b, c],
            [a, c, b],
            [b, a, c],
            [b, c, a],
            [c, a, b],
            [c, b, a]
        ]
        
        best = None
        
        for perm in permutations:
            res = perm[0]
            res = merge(res, perm[1])
            res = merge(res, perm[2])
            if best is None:
                best = res
            else:
                if len(res) < len(best):
                    best = res
                elif len(res) == len(best) and res < best:
                    best = res
        
        return best