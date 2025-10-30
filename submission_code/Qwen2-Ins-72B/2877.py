class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s1, s2):
            m, n = len(s1), len(s2)
            for i in range(n, 0, -1):
                if s2[:i] in s1:
                    return min(s1, s1 + s2[i:], key=lambda x: (len(x), x))
            return min(s1 + s2, s2 + s1, key=lambda x: (len(x), x))
        
        ans = a + b + c
        for x, y, z in [[a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]]:
            ans = merge(merge(x, y), z)
        return ans