class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge_strings(s1, s2):
            # Check if one string is a substring of the other
            if s1 in s2:
                return s2
            if s2 in s1:
                return s1
            
            # Find the maximum overlap between s1 and s2
            max_overlap = min(len(s1), len(s2))
            while max_overlap > 0:
                if s1.endswith(s2[:max_overlap]) or s2.endswith(s1[:max_overlap]):
                    break
                max_overlap -= 1
            
            # Merge s1 and s2 with the maximum overlap
            if s1.endswith(s2[:max_overlap]):
                return s1 + s2[max_overlap:]
            else:
                return s2 + s1[max_overlap:]
        
        # Generate all permutations of a, b, c
        permutations = [
            (a, b, c),
            (a, c, b),
            (b, a, c),
            (b, c, a),
            (c, a, b),
            (c, b, a)
        ]
        
        # Initialize the result with a string that's definitely longer than any possible answer
        result = "z" * (len(a) + len(b) + len(c))
        
        # Try all permutations and keep the lexicographically smallest result
        for perm in permutations:
            merged = merge_strings(perm[0], perm[1])
            merged = merge_strings(merged, perm[2])
            if len(merged) < len(result) or (len(merged) == len(result) and merged < result):
                result = merged
        
        return result