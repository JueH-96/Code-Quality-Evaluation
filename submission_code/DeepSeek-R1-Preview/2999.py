class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Extract and sort characters at even indices (0 and 2)
        s1_evens = sorted([s1[0], s1[2]])
        s2_evens = sorted([s2[0], s2[2]])
        
        # Extract and sort characters at odd indices (1 and 3)
        s1_odds = sorted([s1[1], s1[3]])
        s2_odds = sorted([s2[1], s2[3]])
        
        # Check if both even and odd character sets match
        return s1_evens == s2_evens and s1_odds == s2_odds