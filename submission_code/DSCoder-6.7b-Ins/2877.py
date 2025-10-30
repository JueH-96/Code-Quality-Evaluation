class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def overlap(s1, s2):
            l1, l2 = len(s1), len(s2)
            for l in range(min(l1, l2), 0, -1):
                if s1[l1-l:] == s2[:l]:
                    return s1 + s2[l:]
            return s1 + s2

        ab = overlap(a, b)
        bc = overlap(b, c)
        ac = overlap(a, c)

        res = min(ab, bc, ac, key=len)
        if len(ab) == len(res): res = min(ab, res)
        if len(bc) == len(res): res = min(bc, res)
        if len(ac) == len(res): res = min(ac, res)

        return res