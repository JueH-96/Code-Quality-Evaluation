class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Check if the sorted pairs (0,2) and (1,3) match between s1 and s2
        pair1_s1 = sorted([s1[0], s1[2]])
        pair1_s2 = sorted([s2[0], s2[2]])
        if pair1_s1 != pair1_s2:
            return False
        
        pair2_s1 = sorted([s1[1], s1[3]])
        pair2_s2 = sorted([s2[1], s2[3]])
        if pair2_s1 != pair2_s2:
            return False
        
        return True