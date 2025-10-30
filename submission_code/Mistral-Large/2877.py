class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def is_substring(s1, s2):
            return s1 in s2

        def merge(s1, s2):
            for i in range(len(s2)):
                if is_substring(s2[i:], s1):
                    return s1 + s2[:i]
            return s1 + s2

        def find_min_string(a, b, c):
            candidates = [merge(merge(a, b), c), merge(merge(a, c), b), merge(merge(b, c), a)]
            return min(candidates, key=lambda x: (len(x), x))

        return find_min_string(a, b, c)