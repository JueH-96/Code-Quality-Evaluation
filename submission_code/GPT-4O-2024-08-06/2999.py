class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # We can swap characters at indices (0, 2) and (1, 3)
        # This means characters at even indices can be rearranged among themselves
        # and characters at odd indices can be rearranged among themselves.
        
        # Extract characters at even indices
        even_s1 = [s1[0], s1[2]]
        even_s2 = [s2[0], s2[2]]
        
        # Extract characters at odd indices
        odd_s1 = [s1[1], s1[3]]
        odd_s2 = [s2[1], s2[3]]
        
        # Sort and compare
        return sorted(even_s1) == sorted(even_s2) and sorted(odd_s1) == sorted(odd_s2)