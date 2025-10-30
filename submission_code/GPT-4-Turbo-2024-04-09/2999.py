class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Since the strings are of length 4, we can only swap (0, 2) and (1, 3)
        # Check if s1 is already equal to s2
        if s1 == s2:
            return True
        
        # Check if swapping (0, 2) in s1 makes it equal to s2
        if s1[0] == s2[2] and s1[1] == s2[1] and s1[2] == s2[0] and s1[3] == s2[3]:
            return True
        
        # Check if swapping (1, 3) in s1 makes it equal to s2
        if s1[0] == s2[0] and s1[1] == s2[3] and s1[2] == s2[2] and s1[3] == s2[1]:
            return True
        
        # Check if swapping both (0, 2) and (1, 3) in s1 makes it equal to s2
        if s1[0] == s2[2] and s1[1] == s2[3] and s1[2] == s2[0] and s1[3] == s2[1]:
            return True
        
        # If none of the above conditions are met, return False
        return False