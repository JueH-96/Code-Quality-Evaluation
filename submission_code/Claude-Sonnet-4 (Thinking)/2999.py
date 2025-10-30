class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Characters at positions 0 and 2 can be swapped
        group1_s1 = sorted([s1[0], s1[2]])
        group1_s2 = sorted([s2[0], s2[2]])
        
        # Characters at positions 1 and 3 can be swapped  
        group2_s1 = sorted([s1[1], s1[3]])
        group2_s2 = sorted([s2[1], s2[3]])
        
        # Both groups must have the same characters for strings to be equal
        return group1_s1 == group1_s2 and group2_s1 == group2_s2