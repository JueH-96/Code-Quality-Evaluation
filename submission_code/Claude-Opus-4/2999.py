class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Check if characters at positions 0,2 can be rearranged to match
        if sorted([s1[0], s1[2]]) != sorted([s2[0], s2[2]]):
            return False
        
        # Check if characters at positions 1,3 can be rearranged to match
        if sorted([s1[1], s1[3]]) != sorted([s2[1], s2[3]]):
            return False
            
        return True