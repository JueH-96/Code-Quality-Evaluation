class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge_two(s1, s2):
            # If s2 is already in s1, return s1
            if s2 in s1:
                return s1
            # If s1 is already in s2, return s2
            if s1 in s2:
                return s2
            
            # Find the maximum overlap where end of s1 matches beginning of s2
            for i in range(len(s1), 0, -1):
                if s2.startswith(s1[-i:]):
                    return s1 + s2[i:]
            
            # No overlap, just concatenate
            return s1 + s2
        
        # Try all 6 permutations
        permutations = [
            (a, b, c),
            (a, c, b),
            (b, a, c),
            (b, c, a),
            (c, a, b),
            (c, b, a)
        ]
        
        best = None
        for perm in permutations:
            # Merge the three strings in order
            merged = merge_two(merge_two(perm[0], perm[1]), perm[2])
            
            if best is None or len(merged) < len(best) or (len(merged) == len(best) and merged < best):
                best = merged
        
        return best