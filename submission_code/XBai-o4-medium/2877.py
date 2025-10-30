import itertools

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s, t):
            min_len = min(len(s), len(t))
            max_overlap = 0
            for k in range(min_len, 0, -1):
                if s[-k:] == t[:k]:
                    max_overlap = k
                    break
            return s + t[max_overlap:]
        
        candidates = []
        for perm in itertools.permutations([a, b, c]):
            first_merge = merge(perm[0], perm[1])
            combined = merge(first_merge, perm[2])
            candidates.append(combined)
        
        candidates.sort(key=lambda x: (len(x), x))
        return candidates[0]