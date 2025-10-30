class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations

        # Function to merge two strings with maximum overlap
        def merge(s1, s2):
            max_overlap = 0
            # Check for the largest overlap from the end of s1 and the start of s2
            for overlap in range(1, min(len(s1), len(s2)) + 1):
                if s1[-overlap:] == s2[:overlap]:
                    max_overlap = overlap
            # Merge s1 and s2 with the found overlap
            return s1 + s2[max_overlap:]
        
        # List of input strings
        strings = [a, b, c]
        candidates = []
        
        # Consider all permutations of the three strings
        for perm in permutations(strings):
            # Merge the first two strings
            merged1 = merge(perm[0], perm[1])
            # Merge the result with the third string
            merged_full = merge(merged1, perm[2])
            candidates.append(merged_full)
        
        # Sort candidates by length and then lexicographically
        candidates.sort(key=lambda x: (len(x), x))
        
        # Return the smallest candidate
        return candidates[0]