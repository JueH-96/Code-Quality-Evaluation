import itertools

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s, t):
            if s in t:
                return t
            if t in s:
                return s
            max_overlap_1 = 0
            len_s, len_t = len(s), len(t)
            for k in range(1, min(len_s, len_t) + 1):
                if s[-k:] == t[:k]:
                    max_overlap_1 = k
            merged1 = s + t[max_overlap_1:]
            max_overlap_2 = 0
            for k in range(1, min(len_t, len_s) + 1):
                if t[-k:] == s[:k]:
                    max_overlap_2 = k
            merged2 = t + s[max_overlap_2:]
            if len(merged1) < len(merged2):
                return merged1
            elif len(merged2) < len(merged1):
                return merged2
            else:
                return min(merged1, merged2)
        
        candidates = []
        for perm in itertools.permutations([a, b, c]):
            x, y, z = perm
            s1 = merge(x, y)
            s2 = merge(s1, z)
            candidates.append(s2)
        min_len = min(len(c) for c in candidates)
        min_candidates = [c for c in candidates if len(c) == min_len]
        return min(min_candidates)