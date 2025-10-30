import math # Not needed for this solution, but harmless if left

class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        """
        Calculates the maximum possible length of a concatenated string formed from
        x "AA"s, y "BB"s, and z "AB"s, such that the string does not contain
        "AAA" or "BBB" as substrings.

        Args:
            x: The number of "AA" strings available.
            y: The number of "BB" strings available.
            z: The number of "AB" strings available.

        Returns:
            The maximum possible length of the valid concatenated string.

        Constraints:
            - The concatenated string must not contain "AAA" as a substring.
            - The concatenated string must not contain "BBB" as a substring.

        Analysis:
            - Each block ("AA", "BB", "AB") has a length of 2. The total length
              is 2 times the number of blocks used.
            - The constraint "no AAA" means 'A' cannot appear three times consecutively.
              This implies that an "AA" block cannot be placed immediately after a block
              ending in 'A'. Specifically, "AA" cannot follow "AA" (AAAA) or "BA" (if BA existed).
              Also, "AA" cannot precede "AB" (AAAB). The only valid block to follow "AA" is "BB" (AABB).
            - The constraint "no BBB" means 'B' cannot appear three times consecutively.
              This implies that a "BB" block cannot be placed immediately after a block
              ending in 'B'. Specifically, "BB" cannot follow "BB" (BBBB) or "AB" (ABBB).
              "BB" can follow "AA" (AABB). "BB" can precede "AA" (BBAA) or "AB" (BBAB).
            - The "AB" block ends in 'B'. It can follow "AA" (AAAB - invalid) or "BB" (BBAB - valid) or "AB" (ABAB - valid).
              It starts with 'A'. It can precede "AA" (ABAA - valid) or "BB" (ABBB - invalid) or "AB" (ABAB - valid).

            Summary of valid transitions (Block1 -> Block2):
            - AA -> BB
            - BB -> AA, AB
            - AB -> AA, AB

            Observations:
            - "AA" blocks severely restrict placement. They must be followed by "BB".
            - "BB" blocks are more flexible.
            - "AB" blocks are also quite flexible.

            Structure:
            - The core structure often involves alternating "AA" and "BB" blocks.
              We can use at most min(x, y) pairs of ("AA", "BB").
              This contributes 4 * min(x, y) to the length.
              Example: "AABBAABB..." or "BBAABBAA..."
            - All z "AB" blocks can always be incorporated into the string without
              violating the constraints, contributing 2 * z to the length. They can act
              as connectors or form their own chain ("ABAB...").

            Combining these:
            Let m = min(x, y).
            Base length using m "AA"s, m "BB"s, and z "AB"s:
            Consider a structure like "AABBAA...BB" (m pairs, ends 'B'). We can append
            all z "AB"s ("ABAB...AB", ends 'B'). The combined string "...BB" + "AB..." -> "...BBAB..." is valid.
            The length is 4*m + 2*z.

            Handling the imbalance (|x - y|):
            Case 1: x == y (m = x = y)
              The maximum length is achieved by using all x "AA", y "BB", and z "AB".
              Length = 4*m + 2*z.

            Case 2: x != y
              Let m = min(x, y). There are |x - y| extra blocks of one type.
              If x > y (excess "AA"s):
                Start with "AABBAA...BB" (length 4m, ends 'B'). Uses m AA, m BB.
                Append "ABAB...AB" (length 2z, ends 'B'). Uses z AB.
                String ends in 'B'. We have x - m > 0 "AA"s left.
                We can append one "AA". String becomes "...AB" + "AA" -> "...ABAA". Valid. Length + 2.
                Total length = 4m + 2z + 2. We cannot add more "AA"s (string ends 'A') or "BB"s (none left).
              If y > x (excess "BB"s):
                Start with "BBAABB...AA" (length 4m, ends 'A'). Uses m AA, m BB.
                We have y - m > 0 "BB"s left and z "AB"s.
                String ends in 'A'. We can append "BB". String becomes "...AA" + "BB" -> "...AABB". Valid. Length + 2. Ends 'B'.
                Uses m AA, m+1 BB.
                Now append "ABAB...AB" (length 2z, ends 'B'). String becomes "...AABB" + "AB..." -> "...AABBAB...". Valid.
                Total length = (4m + 2) + 2z = 4m + 2z + 2. We cannot add more "BB"s (string ends 'B').

              In both subcases (x > y or y > x), the maximum length is 4*m + 2*z + 2.

            Combined Formula:
            Let m = min(x, y).
            If x == y, length = 4*m + 2*z.
            If x != y, length = 4*m + 2*z + 2.
        """

        # Determine the number of balanced "AA"-"BB" pairs we can form.
        m = min(x, y)

        # Calculate the maximum length based on whether x and y are equal.
        if x == y:
            # If counts are equal, we use m "AA", m "BB", z "AB".
            # Total length = 4*m (from AA/BB pairs) + 2*z (from AB)
            max_len = 4 * m + 2 * z
        else:
            # If counts differ, we use m "AA", m "BB", z "AB", plus one extra block
            # of the type ("AA" or "BB") that is in excess.
            # Total length = 4*m (from AA/BB pairs) + 2*z (from AB) + 2 (from one extra block)
            max_len = 4 * m + 2 * z + 2

        return max_len