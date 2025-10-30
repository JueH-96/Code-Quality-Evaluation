class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Check if the characters at even indices can be made equal
        if sorted(s1[::2]) == sorted(s2[::2]) and sorted(s1[1::2]) == sorted(s2[1::2]):
            return True
        return False