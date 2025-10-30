class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Extract even and odd indexed characters for both strings
        s1_even = sorted([s1[0], s1[2]])
        s1_odd = sorted([s1[1], s1[3]])
        s2_even = sorted([s2[0], s2[2]])
        s2_odd = sorted([s2[1], s2[3]])
        
        # Check if the sorted even and odd parts are equal between s1 and s2
        return s1_even == s2_even and s1_odd == s2_odd