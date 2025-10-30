class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Convert strings to lists for easier manipulation
        s1_list = list(s1)
        s2_list = list(s2)
        
        # Check if strings are already equal
        if s1_list == s2_list:
            return True
            
        # Try all possible combinations of swaps with j-i=2
        # We only need to try each swap once since order doesn't matter
        # For length 4, possible swaps are (0,2) and (1,3)
        
        # Try first possible swap (0,2)
        s1_try1 = s1_list.copy()
        s1_try1[0], s1_try1[2] = s1_try1[2], s1_try1[0]
        if s1_try1 == s2_list:
            return True
            
        # Try second possible swap (1,3)
        s1_try2 = s1_list.copy()
        s1_try2[1], s1_try2[3] = s1_try2[3], s1_try2[1]
        if s1_try2 == s2_list:
            return True
            
        # Try both swaps
        s1_try3 = s1_try1.copy()
        s1_try3[1], s1_try3[3] = s1_try3[3], s1_try3[1]
        if s1_try3 == s2_list:
            return True
            
        return False