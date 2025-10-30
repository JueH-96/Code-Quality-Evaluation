class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Check if the sorted strings are equal
        return sorted(s1) == sorted(s2)