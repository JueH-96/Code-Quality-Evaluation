class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s1, s2):
            if s1 in s2:
                return s2
            if s2 in s1:
                return s1
            for i in range(len(s1)):
                if s2.startswith(s1[i:]):
                    return s1[:i] + s2
            return s1 + s2
        
        candidates = [
            merge(merge(a, b), c),
            merge(merge(a, c), b),
            merge(merge(b, a), c),
            merge(merge(b, c), a),
            merge(merge(c, a), b),
            merge(merge(c, b), a)
        ]
        
        candidates.sort(key=lambda x: (len(x), x))
        return candidates[0]