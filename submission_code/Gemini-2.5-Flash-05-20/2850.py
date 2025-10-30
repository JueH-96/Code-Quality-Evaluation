class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        """
        Calculates the maximum possible length of a string formed by concatenating
        x "AA" strings, y "BB" strings, and z "AB" strings, such that the
        resulting string does not contain "AAA" or "BBB" as a substring.

        The problem effectively boils down to finding the maximum length of a
        string composed of individual 'A' and 'B' characters where no three
        consecutive characters are the same.

        Let:
        total_A = Total count of 'A' characters available.
                  Each "AA" contributes 2 'A's. Each "AB" contributes 1 'A'.
                  So, total_A = 2 * x + z
        total_B = Total count of 'B' characters available.
                  Each "BB" contributes 2 'B's. Each "AB" contributes 1 'B'.
                  So, total_B = 2 * y + z

        The maximum length of such a string is achieved by alternating 'A's and 'B's.
        If total_A equals total_B, we can form a perfectly alternating string
        like "AABBAABBAABB..." or "ABABAB...". The length would be 2 * total_A (or 2 * total_B).
        
        If total_A is not equal to total_B, we can still form an almost perfectly
        alternating string, but one character type will have one more block of
        two characters (e.g., "AABBAABBAA"). This adds 2 to the length compared
        to the balanced case.
        
        The general formula for the maximum length is:
        2 * min(total_A, total_B) + (2 if total_A != total_B else 0)

        The specific block types "AA", "BB", "AB" are flexible enough to construct
        this optimal character sequence because:
        - "AA" and "BB" provide segments of two identical characters.
        - "AB" provides an 'A' followed by a 'B', acting as a bridge or standalone segment.
          It can follow a 'B' (e.g., "BBAB") or precede an 'A' (e.g., "ABAA"),
          and can follow itself (e.g., "ABAB").
          The only restrictions are to prevent "AAA" or "BBB" from forming due to block boundaries:
          - "AA" cannot be followed by "AA" or "AB".
          - "BB" cannot be followed by "BB".
          - "AB" cannot be followed by "BB".
        These restrictions still allow for the construction of the character-level optimal string.
        """
        
        total_a_chars = 2 * x + z
        total_b_chars = 2 * y + z
        
        # Calculate the base length from alternating segments of 'A' and 'B'
        # Each pair of (A_segment, B_segment) contributes 4 characters (e.g., AABB)
        # We can form min(total_a_chars, total_b_chars) such alternating segments.
        max_len = 2 * min(total_a_chars, total_b_chars)
        
        # If the counts are not equal, we can add two more characters from the majority type.
        # For example, if total_A = 6 and total_B = 4, we can form "AABBAABBAA" (length 10).
        # This is 2 * min(6,4) + 2 = 2 * 4 + 2 = 10.
        if total_a_chars != total_b_chars:
            max_len += 2
            
        return max_len