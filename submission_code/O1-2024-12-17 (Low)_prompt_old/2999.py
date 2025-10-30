class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Possible transformations of s1 by the given swap operations:
        # - No swap
        if s1 == s2:
            return True
        
        # - Swap indices (0,2)
        t1 = s1[2] + s1[1] + s1[0] + s1[3]
        if t1 == s2:
            return True
        
        # - Swap indices (1,3)
        t2 = s1[0] + s1[3] + s1[2] + s1[1]
        if t2 == s2:
            return True
        
        # - Swap indices (0,2) then (1,3) (order doesn't matter since they commute)
        t3 = t1[0] + t1[3] + t1[2] + t1[1]  # or from s1: s1[2] + s1[3] + s1[0] + s1[1]
        if t3 == s2:
            return True
        
        return False