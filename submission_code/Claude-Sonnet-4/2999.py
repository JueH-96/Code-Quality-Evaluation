class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Check if characters at positions 0,2 in s1 can match positions 0,2 in s2
        # Check if characters at positions 1,3 in s1 can match positions 1,3 in s2
        
        # Group 1: positions 0 and 2
        group1_s1 = sorted([s1[0], s1[2]])
        group1_s2 = sorted([s2[0], s2[2]])
        
        # Group 2: positions 1 and 3
        group2_s1 = sorted([s1[1], s1[3]])
        group2_s2 = sorted([s2[1], s2[3]])
        
        return group1_s1 == group1_s2 and group2_s1 == group2_s2