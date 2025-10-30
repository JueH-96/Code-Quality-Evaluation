import itertools

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        perms = list(itertools.permutations([a, b, c]))
        best = None
        for p in perms:
            temp = self.merge_two(p[0], p[1])
            candidate = self.merge_two(temp, p[2])
            if best is None:
                best = candidate
            else:
                if len(candidate) < len(best):
                    best = candidate
                elif len(candidate) == len(best) and candidate < best:
                    best = candidate
        return best

    def merge_two(self, s, t):
        if t in s:
            return s
        n = min(len(s), len(t))
        for i in range(n, -1, -1):
            if s.endswith(t[:i]):
                return s + t[i:]
        return s + t