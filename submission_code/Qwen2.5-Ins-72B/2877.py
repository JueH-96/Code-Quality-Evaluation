class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s1, s2):
            if s1 in s2:
                return s2
            if s2 in s1:
                return s1
            for i in range(len(s1), 0, -1):
                if s2.startswith(s1[-i:]):
                    return s1 + s2[i:]
            return s1 + s2

        def combine(s1, s2, s3):
            candidates = [
                merge(merge(s1, s2), s3),
                merge(merge(s1, s3), s2),
                merge(merge(s2, s1), s3),
                merge(merge(s2, s3), s1),
                merge(merge(s3, s1), s2),
                merge(merge(s3, s2), s1),
            ]
            return min(candidates, key=lambda x: (len(x), x))

        return combine(a, b, c)