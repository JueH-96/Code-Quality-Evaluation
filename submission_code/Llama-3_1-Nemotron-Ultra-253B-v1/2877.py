import itertools

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        # Check if any of the strings contains all three
        candidates = []
        for s in [a, b, c]:
            if a in s and b in s and c in s:
                candidates.append(s)
        if candidates:
            return min(candidates)
        
        # Generate all permutations of the three strings
        perms = itertools.permutations([a, b, c])
        merged_candidates = []
        
        def merge(s, t):
            if t in s:
                return s
            max_overlap = 0
            len_s = len(s)
            len_t = len(t)
            for i in range(1, min(len_s, len_t) + 1):
                if s.endswith(t[:i]):
                    max_overlap = i
            return s + t[max_overlap:]
        
        for perm in perms:
            s1, s2, s3 = perm
            merged_12 = merge(s1, s2)
            merged_123 = merge(merged_12, s3)
            merged_candidates.append(merged_123)
        
        # Find the minimal length
        min_length = min(len(s) for s in merged_candidates)
        # Filter those with minimal length and find lex smallest
        min_candidates = [s for s in merged_candidates if len(s) == min_length]
        return min(min_candidates)