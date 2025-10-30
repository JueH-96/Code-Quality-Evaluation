class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Identity (no swap)
        s_id = s1
        
        # Swap indices 0 and 2
        s_p1 = s1[2] + s1[1] + s1[0] + s1[3]
        
        # Swap indices 1 and 3
        s_p2 = s1[0] + s1[3] + s1[2] + s1[1]
        
        # Swap indices 0 and 2, then 1 and 3
        s_p1p2 = s1[2] + s1[3] + s1[0] + s1[1]
        
        return s2 in {s_id, s_p1, s_p2, s_p1p2}