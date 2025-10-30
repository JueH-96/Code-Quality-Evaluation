class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        """
        Calculates the maximum possible length of a string formed by concatenating
        "AA", "BB", and "AB" strings, without creating "AAA" or "BBB" substrings.

        Args:
          x: The number of available "AA" strings.
          y: The number of available "BB" strings.
          z: The number of available "AB" strings.

        Returns:
          The maximum possible length of the constructed string.
        """

        # The length of each building block ("AA", "BB", "AB") is 2.
        # To maximize the total length, we need to maximize the number of blocks used.

        # 1. "AB" blocks:
        # We can always use all 'z' of the "AB" blocks by concatenating them: "ABAB...AB".
        # This chain is valid and contributes 'z' blocks.

        # 2. "AA" and "BB" blocks:
        # To avoid "AAA" and "BBB", "AA" and "BB" blocks must be placed alternately.
        # In such a sequence, the count of "AA" blocks and "BB" blocks can differ by at most 1.
        # We want to maximize the number of "AA" and "BB" blocks used under this constraint.

        # - If x == y: We can use all 'x' "AA"s and 'y' "BB"s. Total AA/BB blocks: x + y = 2*x.
        # - If x != y: Let m = min(x, y). We can use 'm' blocks of the less numerous type
        #   and 'm+1' of the more numerous type. Total AA/BB blocks: m + (m+1) = 2*m + 1.

        # 3. Combining:
        # A chain of "AB"s can always be joined with an alternating chain of "AA"s and "BB"s.
        # Therefore, the total number of blocks is the sum from both parts.
        
        if x == y:
            # All 'z' ABs, 'x' AAs, and 'y' BBs can be used.
            # Total blocks = z + x + y = z + 2*x.
            total_blocks = z + 2 * x
        else:
            # Let m = min(x, y). We use z "AB"s, and an alternating chain of m and m+1 blocks.
            # Total blocks = z + m + (m+1) = z + 2*m + 1.
            m = min(x, y)
            total_blocks = z + 2 * m + 1
        
        # Each block has a length of 2.
        return total_blocks * 2