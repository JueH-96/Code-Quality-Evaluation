class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations
        
        # Function to merge two strings with minimum overlap
        def merge(s1, s2):
            # Check for the maximum overlap
            max_overlap = 0
            for i in range(1, min(len(s1), len(s2)) + 1):
                if s1[-i:] == s2[:i]:
                    max_overlap = i
            return s1 + s2[max_overlap:]
        
        # Generate all permutations of the strings
        strings = [a, b, c]
        min_string = None
        
        for perm in permutations(strings):
            # Merge the three strings in the order of the permutation
            merged = merge(merge(perm[0], perm[1]), perm[2])
            # Update the minimum string found
            if min_string is None or len(merged) < len(min_string) or (len(merged) == len(min_string) and merged < min_string):
                min_string = merged
        
        return min_string