class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Since we can swap characters at indices (0, 2) and (1, 3),
        # we can group the characters into two pairs.
        # The first pair consists of characters at indices 0 and 2,
        # and the second pair consists of characters at indices 1 and 3.
        
        # Group characters of s1
        s1_pair1 = sorted([s1[0], s1[2]])
        s1_pair2 = sorted([s1[1], s1[3]])
        
        # Group characters of s2
        s2_pair1 = sorted([s2[0], s2[2]])
        s2_pair2 = sorted([s2[1], s2[3]])
        
        # Check if both pairs match
        return s1_pair1 == s2_pair1 and s1_pair2 == s2_pair2