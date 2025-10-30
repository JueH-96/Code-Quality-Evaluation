class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        """
        Determines if two strings of length 4 can be made equal using specific swap operations.

        The allowed operations for a string 's' of length 4 are:
        1. Swap s[0] and s[2].
        2. Swap s[1] and s[3].

        These operations are independent. For s1 and s2 to be made equal:
        - The characters at even positions (0 and 2) in s1 must be the same as
          the characters at even positions (0 and 2) in s2, ignoring order.
        - The characters at odd positions (1 and 3) in s1 must be the same as
          the characters at odd positions (1 and 3) in s2, ignoring order.

        We can check this by sorting the characters for each pair of indices
        and then comparing the sorted pairs.
        """

        # Extract and sort characters from even positions for s1 and s2
        # s1[0] and s1[2] form one group that can be swapped
        even_chars_s1 = sorted([s1[0], s1[2]])
        even_chars_s2 = sorted([s2[0], s2[2]])

        # Extract and sort characters from odd positions for s1 and s2
        # s1[1] and s1[3] form another independent group that can be swapped
        odd_chars_s1 = sorted([s1[1], s1[3]])
        odd_chars_s2 = sorted([s2[1], s2[3]])

        # Both groups must match for the strings to be transformable into each other
        return (even_chars_s1 == even_chars_s2) and \
               (odd_chars_s1 == odd_chars_s2)