class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # The allowed operations let us swap characters at positions (0,2) or (1,3).
        # These operations generate at most 4 possible permutations of s1:
        #   1) s1 itself
        #   2) swap s1[0] and s1[2]
        #   3) swap s1[1] and s1[3]
        #   4) swap both (0,2) and (1,3)
        # We just check if s2 matches any of them.
        
        possible_strings = set()
        
        # Original string
        possible_strings.add(s1)
        
        # Swap positions (0,2)
        s1_swap_02 = s1[2] + s1[1] + s1[0] + s1[3]
        possible_strings.add(s1_swap_02)
        
        # Swap positions (1,3)
        s1_swap_13 = s1[0] + s1[3] + s1[2] + s1[1]
        possible_strings.add(s1_swap_13)
        
        # Swap both (0,2) and (1,3)
        # We can apply either swap_02 then swap_13 or vice versa
        s1_swap_02_13 = s1_swap_02[0] + s1_swap_02[3] + s1_swap_02[2] + s1_swap_02[1]
        # Alternatively:
        # s1_swap_13_02 = s1_swap_13[2] + s1_swap_13[1] + s1_swap_13[0] + s1_swap_13[3]
        # Both results are the same, so just do one.
        possible_strings.add(s1_swap_02_13)
        
        return s2 in possible_strings