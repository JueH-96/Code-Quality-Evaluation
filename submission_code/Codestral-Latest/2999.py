class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Check if the strings are already equal
        if s1 == s2:
            return True

        # Check if swapping the middle two characters makes the strings equal
        if s1[0] == s2[0] and s1[3] == s2[3] and s1[1] == s2[3] and s1[2] == s2[1]:
            return True

        # Check if swapping the first two characters and the last two characters makes the strings equal
        if s1[0] == s2[2] and s1[1] == s2[3] and s1[2] == s2[0] and s1[3] == s2[1]:
            return True

        # If none of the above conditions are met, the strings cannot be made equal
        return False