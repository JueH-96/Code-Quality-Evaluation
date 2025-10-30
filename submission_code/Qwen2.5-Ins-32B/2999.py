class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Check if characters at even indices can be swapped to match
        if sorted(s1[::2]) != sorted(s2[::2]):
            return False
        
        # Check if characters at odd indices can be swapped to match
        if sorted(s1[1::2]) != sorted(s2[1::2]):
            return False
        
        return True