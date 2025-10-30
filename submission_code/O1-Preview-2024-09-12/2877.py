class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations

        def overlap(s1, s2):
            if s2 in s1:
                return s1
            if s1 in s2:
                return s2
            max_len = min(len(s1), len(s2))
            for i in range(max_len, 0, -1):
                if s1[-i:] == s2[:i]:
                    return s1 + s2[i:]
            return s1 + s2

        strings = [a, b, c]
        candidates = []
        for perm in permutations(strings):
            s1, s2, s3 = perm
            # Merge s1 and s2 in both possible overlaps
            m1 = overlap(s1, s2)
            m2 = overlap(s2, s1)
            for m in [m1, m2]:
                # Merge the result with s3 in both possible overlaps
                m3a = overlap(m, s3)
                m3b = overlap(s3, m)
                for final in [m3a, m3b]:
                    if a in final and b in final and c in final:
                        candidates.append(final)
        if not candidates:
            return a + b + c
        min_len = min(len(s) for s in candidates)
        min_strings = [s for s in candidates if len(s) == min_len]
        return min(min_strings)