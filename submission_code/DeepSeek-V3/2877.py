class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations
        
        def merge(s1, s2):
            # Find the maximum overlap between s1 and s2
            max_len = 0
            for i in range(1, min(len(s1), len(s2)) + 1):
                if s1[-i:] == s2[:i]:
                    max_len = i
            return s1 + s2[max_len:]
        
        # Generate all possible permutations of a, b, c
        candidates = []
        for perm in permutations([a, b, c]):
            s1, s2, s3 = perm
            merged = merge(merge(s1, s2), s3)
            candidates.append(merged)
        
        # Find the candidate with the minimum length
        min_len = min(len(candidate) for candidate in candidates)
        # Filter candidates with the minimum length
        min_candidates = [candidate for candidate in candidates if len(candidate) == min_len]
        # Return the lex smallest one
        return min(min_candidates)