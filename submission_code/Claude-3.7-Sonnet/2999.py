class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Original string
        if s1 == s2:
            return True
        
        # After swapping (0,2)
        if s1[2] + s1[1] + s1[0] + s1[3] == s2:
            return True
        
        # After swapping (1,3)
        if s1[0] + s1[3] + s1[2] + s1[1] == s2:
            return True
        
        # After swapping both (0,2) and (1,3)
        if s1[2] + s1[3] + s1[0] + s1[1] == s2:
            return True
        
        return False