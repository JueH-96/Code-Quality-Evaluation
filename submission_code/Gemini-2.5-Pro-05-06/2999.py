class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Check if characters at even positions (0 and 2) can match.
        # The pair (s1[0], s1[2]) must be a permutation of (s2[0], s2[2]).
        # This is true if (s1[0] == s2[0] AND s1[2] == s2[2]) OR 
        # (s1[0] == s2[2] AND s1[2] == s2[0]).
        condition_even_indices = (s1[0] == s2[0] and s1[2] == s2[2]) or \
                                 (s1[0] == s2[2] and s1[2] == s2[0])
        
        # Check if characters at odd positions (1 and 3) can match.
        # The pair (s1[1], s1[3]) must be a permutation of (s2[1], s2[3]).
        # This is true if (s1[1] == s2[1] AND s1[3] == s2[3]) OR 
        # (s1[1] == s2[3] AND s1[3] == s2[1]).
        condition_odd_indices = (s1[1] == s2[1] and s1[3] == s2[3]) or \
                                (s1[1] == s2[3] and s1[3] == s2[1])
                                
        # Both conditions must be met for s1 to be transformable into s2.
        return condition_even_indices and condition_odd_indices