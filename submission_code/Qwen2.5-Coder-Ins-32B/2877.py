class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(x: str, y: str) -> str:
            if y in x:
                return x
            for i in range(len(y), -1, -1):
                if x.endswith(y[:i]):
                    return x + y[i:]
            return x + y

        def find_min_string(s1: str, s2: str, s3: str) -> str:
            candidates = [
                merge(merge(s1, s2), s3),
                merge(merge(s1, s3), s2),
                merge(merge(s2, s1), s3),
                merge(merge(s2, s3), s1),
                merge(merge(s3, s1), s2),
                merge(merge(s3, s2), s1)
            ]
            return min(candidates, key=lambda s: (len(s), s))

        return find_min_string(a, b, c)