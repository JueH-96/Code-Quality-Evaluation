class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Sort the strings and compare them
        return sorted(s1) == sorted(s2)