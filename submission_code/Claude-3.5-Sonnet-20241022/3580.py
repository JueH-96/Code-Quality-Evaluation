class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        
        if m > n:
            return -1
            
        # Helper function to check if two strings are almost equal
        def isAlmostEqual(s1: str, s2: str) -> bool:
            if len(s1) != len(s2):
                return False
                
            diff = 0
            for c1, c2 in zip(s1, s2):
                if c1 != c2:
                    diff += 1
                if diff > 1:
                    return False
            return True
        
        # Check each possible substring of length m
        for i in range(n - m + 1):
            if isAlmostEqual(s[i:i+m], pattern):
                return i
                
        return -1