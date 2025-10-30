class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        """
        Determines if s1 can be made equal to s2 using the allowed swap operation.

        The problem states we can swap characters at indices i and j if j - i = 2.
        For a 4-character string s = c0 c1 c2 c3, the indices are 0, 1, 2, 3.
        The possible pairs (i, j) such that j - i = 2 are:
        - i = 0, j = 2
        - i = 1, j = 3

        This means we can perform the following swaps any number of times:
        - Swap the character at index 0 with the character at index 2.
        - Swap the character at index 1 with the character at index 3.

        Notice that these two swap operations are independent. The characters at even positions (0 and 2)
        can only be swapped with each other. The characters at odd positions (1 and 3) can only be
        swapped with each other.

        Therefore, string s1 can be transformed into string s2 if and only if:
        1. The set of characters at even indices in s1 ({s1[0], s1[2]}) is the same as the set of
           characters at even indices in s2 ({s2[0], s2[2]}). Since these are pairs, this means
           either (s1[0] == s2[0] and s1[2] == s2[2]) or (s1[0] == s2[2] and s1[2] == s2[0]).
        2. The set of characters at odd indices in s1 ({s1[1], s1[3]}) is the same as the set of
           characters at odd indices in s2 ({s2[1], s2[3]}). Similarly, this means
           either (s1[1] == s2[1] and s1[3] == s2[3]) or (s1[1] == s2[3] and s1[3] == s2[1]).

        Both conditions must be true for the strings to be transformable into each other.

        Args:
          s1: The first input string of length 4.
          s2: The second input string of length 4.

        Returns:
          True if s1 can be made equal to s2, False otherwise.
        """

        # Check if the characters at even indices (0, 2) match as a multiset.
        # This condition checks if {s1[0], s1[2]} == {s2[0], s2[2]}.
        even_indices_match = (s1[0] == s2[0] and s1[2] == s2[2]) or \
                             (s1[0] == s2[2] and s1[2] == s2[0])

        # If the characters at even indices cannot form the same pair,
        # then s1 cannot be transformed into s2. We can return False early.
        if not even_indices_match:
            return False

        # Check if the characters at odd indices (1, 3) match as a multiset.
        # This condition checks if {s1[1], s1[3]} == {s2[1], s2[3]}.
        odd_indices_match = (s1[1] == s2[1] and s1[3] == s2[3]) or \
                            (s1[1] == s2[3] and s1[3] == s2[1])

        # Both conditions (even and odd index character sets matching) must be true.
        # Since we already checked that even_indices_match is true, the final result
        # depends only on whether odd_indices_match is also true.
        return odd_indices_match