class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Since we can only swap characters at indices with a difference of 2,
        # we can only swap the first with the third and the second with the fourth.
        # Therefore, if s1 can be transformed into s2, the characters at the
        # first and third positions must be the same in both strings (though not necessarily in the same order),
        # and the same applies to the second and fourth positions.
        
        # Check if the sorted characters at positions 1 and 3 are the same for both strings
        if sorted(s1[0] + s1[2]) != sorted(s2[0] + s2[2]):
            return False
        
        # Check if the sorted characters at positions 2 and 4 are the same for both strings
        if sorted(s1[1] + s1[3]) != sorted(s2[1] + s2[3]):
            return False
        
        # If both conditions are satisfied, s1 can be transformed into s2
        return True