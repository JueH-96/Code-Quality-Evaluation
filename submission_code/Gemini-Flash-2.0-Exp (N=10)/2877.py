class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s1, s2):
            for i in range(len(s1), 0, -1):
                if s2.startswith(s1[-i:]):
                    return s1 + s2[i:]
            return s1 + s2

        import itertools
        perms = list(itertools.permutations([a, b, c]))
        
        min_len = float('inf')
        ans = ""

        for p in perms:
            merged = merge(merge(p[0], p[1]), p[2])
            if len(merged) < min_len:
                min_len = len(merged)
                ans = merged
            elif len(merged) == min_len and merged < ans:
                ans = merged
        
        return ans