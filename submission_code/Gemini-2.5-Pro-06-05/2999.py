class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        """
        Determines if s1 can be made equal to s2 by swapping characters
        at indices i and j where j - i = 2.
        """
        
        # The operation allows swapping s[0] with s[2], and s[1] with s[3].
        # This partitions the string's indices into two independent pairs:
        # the even indices {0, 2} and the odd indices {1, 3}.
        
        # For s1 to be transformable into s2, the multiset of characters at the even
        # indices of s1 must be the same as that of s2. The same must hold for the odd indices.
        
        # We can check for multiset equality of two elements {a, b} and {c, d} by checking
        # if (a==c and b==d) or (a==d and b==c).

        # Check if the characters at even indices {0, 2} form the same multiset.
        even_indices_match = (s1[0] == s2[0] and s1[2] == s2[2]) or \
                             (s1[0] == s2[2] and s1[2] == s2[0])
        
        # Check if the characters at odd indices {1, 3} form the same multiset.
        odd_indices_match = (s1[1] == s2[1] and s1[3] == s2[3]) or \
                            (s1[1] == s2[3] and s1[3] == s2[1])
        
        # Both conditions must be true for the strings to be transformable into each other.
        return even_indices_match and odd_indices_match