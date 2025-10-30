from itertools import permutations

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        candidates = []
        for p in permutations([a, b, c]):
            s1, s2, s3 = p
            merged = self.merge_two(s1, s2)
            final = self.merge_two(merged, s3)
            candidates.append(final)
        
        min_length = min(len(c) for c in candidates)
        min_candidates = [c for c in candidates if len(c) == min_length]
        return min(min_candidates)
    
    def merge_two(self, s1: str, s2: str) -> str:
        if s2 in s1:
            return s1
        if s1 in s2:
            return s2
        
        max_overlap = 0
        for i in range(1, min(len(s1), len(s2)) + 1):
            if s1.endswith(s2[:i]):
                max_overlap = i
        candidate1 = s1 + s2[max_overlap:]
        
        max_overlap_rev = 0
        for i in range(1, min(len(s1), len(s2)) + 1):
            if s2.endswith(s1[:i]):
                max_overlap_rev = i
        candidate2 = s2 + s1[max_overlap_rev:]
        
        if len(candidate1) < len(candidate2):
            return candidate1
        elif len(candidate2) < len(candidate1):
            return candidate2
        else:
            return candidate1 if candidate1 < candidate2 else candidate2