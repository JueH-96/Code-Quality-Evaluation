class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s1, s2):
            # Try to merge s2 into s1 by finding the maximum overlap
            # First check if s2 is already a substring of s1
            if s2 in s1:
                return s1
            
            # Find the maximum overlap where end of s1 matches beginning of s2
            max_overlap = 0
            for i in range(1, min(len(s1), len(s2)) + 1):
                if s1[-i:] == s2[:i]:
                    max_overlap = i
            
            return s1 + s2[max_overlap:]
        
        def merge_three(x, y, z):
            # Merge three strings in order
            temp = merge(x, y)
            return merge(temp, z)
        
        # Try all 6 permutations of a, b, c
        candidates = []
        strings = [a, b, c]
        
        # Generate all permutations
        import itertools
        for perm in itertools.permutations(strings):
            result = merge_three(perm[0], perm[1], perm[2])
            candidates.append(result)
        
        # Find the shortest string, and if there are ties, the lexicographically smallest
        min_len = min(len(s) for s in candidates)
        shortest_candidates = [s for s in candidates if len(s) == min_len]
        
        return min(shortest_candidates)