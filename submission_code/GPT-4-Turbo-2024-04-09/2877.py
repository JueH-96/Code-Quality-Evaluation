class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations
        
        def overlap(s1, s2):
            """ Returns the maximum overlap of s1 ending and s2 starting """
            max_overlap = 0
            # Check all possible overlaps from full s1 to just one character
            for i in range(1, min(len(s1), len(s2)) + 1):
                if s1[-i:] == s2[:i]:
                    max_overlap = i
            return max_overlap
        
        def merge(s1, s2):
            """ Merge two strings with maximum overlap """
            max_overlap = overlap(s1, s2)
            return s1 + s2[max_overlap:]
        
        # Try all permutations of the strings to find the minimum length string
        strings = [a, b, c]
        min_string = None
        
        for perm in permutations(strings):
            # Merge according to the permutation
            merged = merge(merge(perm[0], perm[1]), perm[2])
            if min_string is None or len(merged) < len(min_string) or (len(merged) == len(min_string) and merged < min_string):
                min_string = merged
        
        return min_string