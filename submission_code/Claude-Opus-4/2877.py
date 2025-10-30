class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s1, s2):
            # If s1 contains s2, return s1
            if s2 in s1:
                return s1
            # If s2 contains s1, return s2
            if s1 in s2:
                return s2
            
            # Find the maximum overlap between end of s1 and start of s2
            max_overlap = 0
            for i in range(1, min(len(s1), len(s2)) + 1):
                if s1[-i:] == s2[:i]:
                    max_overlap = i
            
            return s1 + s2[max_overlap:]
        
        def merge_three(x, y, z):
            # Merge x and y first, then merge with z
            temp = merge(x, y)
            return merge(temp, z)
        
        # Try all 6 permutations
        candidates = [
            merge_three(a, b, c),
            merge_three(a, c, b),
            merge_three(b, a, c),
            merge_three(b, c, a),
            merge_three(c, a, b),
            merge_three(c, b, a)
        ]
        
        # Find the minimum length
        min_len = min(len(s) for s in candidates)
        
        # Filter candidates with minimum length and return lexicographically smallest
        min_candidates = [s for s in candidates if len(s) == min_len]
        return min(min_candidates)