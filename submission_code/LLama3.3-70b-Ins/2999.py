class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # If the two strings are not anagrams of each other, they can never be equal
        if sorted(s1) != sorted(s2):
            return False
        
        # If the strings are anagrams, we can try to make them equal by swapping characters
        # We only need to check if the characters at indices 0 and 2, and 1 and 3 are the same
        # If they are, we can swap them to make the strings equal
        return (s1[0] == s2[0] and s1[1] == s2[1]) or (s1[0] == s2[2] and s1[2] == s2[0]) or (s1[1] == s2[3] and s1[3] == s2[1]) or (s1[0] == s2[2] and s1[1] == s2[3] and s1[2] == s2[0] and s1[3] == s2[1])