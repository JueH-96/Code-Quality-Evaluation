class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations
        
        # Helper function to merge two strings with maximum overlap.
        def merge_two_strings(s1: str, s2: str) -> str:
            # If s2 is already a substring of s1, no need to add anything.
            if s2 in s1:
                return s1
            
            # Find the maximum overlap where the suffix of s1 matches the prefix of s2.
            max_overlap = 0
            min_len = min(len(s1), len(s2))
            for i in range(1, min_len + 1):
                if s1.endswith(s2[:i]):
                    max_overlap = i
            
            # Merge them by overlapping the matching part.
            return s1 + s2[max_overlap:]
        
        # Try all permutations of a, b, c.
        # For each permutation, merge them one by one and track the result.
        candidates = []
        for x, y, z in permutations([a, b, c]):
            merged_xy = merge_two_strings(x, y)
            merged_xyz = merge_two_strings(merged_xy, z)
            candidates.append(merged_xyz)
        
        # From all candidates, choose the one with minimum length
        # and among them the lexicographically smallest.
        answer = min(candidates, key=lambda s: (len(s), s))
        return answer