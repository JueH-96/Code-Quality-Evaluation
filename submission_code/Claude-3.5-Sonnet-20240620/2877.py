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

        def combine(x, y, z):
            return merge(merge(x, y), z)

        candidates = [
            combine(a, b, c),
            combine(a, c, b),
            combine(b, a, c),
            combine(b, c, a),
            combine(c, a, b),
            combine(c, b, a)
        ]

        return min(candidates, key=lambda x: (len(x), x))