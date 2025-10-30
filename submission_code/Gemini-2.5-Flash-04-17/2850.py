class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # We want to maximize the number of blocks (AA, BB, AB) used, say k_x, k_y, k_z.
        # Total length = 2 * (k_x + k_y + k_z). Maximize num_blocks = k_x + k_y + k_z.
        # Constraints: 0 <= k_x <= x, 0 <= k_y <= y, 0 <= k_z <= z.
        # Structural constraints from "AAA", "BBB" forbidden substrings:
        # - A "BB" block cannot be preceded by a "BB" block or an "AB" block.
        #   This implies a "BB" block must be preceded by an "AA" block or be the first block.
        #   The number of "AA" blocks used must be >= (number of "BB" blocks used) - 1.
        #   k_x >= k_y - 1

        # - An "AA" block cannot be preceded by an "AA" block.
        #   This implies an "AA" block must be preceded by a "BB" block or an "AB" block or be the first block.
        #   The number of "BB" blocks + "AB" blocks used must be >= (number of "AA" blocks used) - 1.
        #   k_y + k_z >= k_x - 1

        # We want to maximize k_x + k_y + k_z subject to:
        # 1. 0 <= k_x <= x
        # 2. 0 <= k_y <= y
        # 3. 0 <= k_z <= z
        # 4. k_x >= k_y - 1  <=> k_y <= k_x + 1
        # 5. k_y + k_z >= k_x - 1 <=> k_x <= k_y + k_z + 1

        # We can use all z AB blocks. So k_z = z. We maximize k_x + k_y + z.
        # This is equivalent to maximizing k_x + k_y subject to:
        # 1. 0 <= k_x <= x
        # 2. 0 <= k_y <= y
        # 3. k_y - 1 <= k_x <= k_y + z + 1

        # The feasible region for (k_x, k_y) is the intersection of the rectangle [0, x] x [0, y]
        # and the band k_y - 1 <= k_x <= k_y + z + 1.
        # We want to maximize the sum k_x + k_y within this region. The maximum occurs at a vertex of the feasible region.
        # The vertices are formed by intersections of boundary lines:
        # k_x = 0, k_x = x, k_y = 0, k_y = y, k_x = k_y - 1, k_x = k_y + z + 1.

        # Consider the top-right corner of the rectangle (x, y).
        # If (x, y) is within or on the boundary of the band k_y - 1 <= k_x <= k_y + z + 1,
        # then (x, y) maximizes k_x + k_y within the rectangle subject to the band constraints.
        # This occurs when y - 1 <= x <= y + z + 1.
        # In this case, we can use k_x = x and k_y = y. Total blocks = x + y + z.

        # If x < y - 1: The rectangle is to the "left" of the band's lower bound k_x = k_y - 1.
        # The feasible region is bounded by k_x <= x, k_y <= y, k_x >= 0, k_y >= 0, and k_x = k_y - 1.
        # To maximize k_x + k_y = (k_y - 1) + k_y = 2k_y - 1, we maximize k_y.
        # k_y is limited by k_y <= y and k_x >= k_y - 1 => x >= k_y - 1 => k_y <= x + 1.
        # So max k_y = min(y, x + 1).
        # Since x < y - 1, we have x + 1 <= y - 1 < y. So min(y, x + 1) = x + 1.
        # Max k_y = x + 1. Max k_x = k_y - 1 = x.
        # Total blocks = k_x + k_y + k_z = x + (x + 1) + z = 2x + 1 + z.
        # This is valid if k_x <= x (x <= x ok) and k_y <= y (x+1 <= y ok, as x < y-1).

        # If x > y + z + 1: The rectangle is to the "right" of the band's upper bound k_x = k_y + z + 1.
        # The feasible region is bounded by k_x <= x, k_y <= y, k_x >= 0, k_y >= 0, and k_x = k_y + z + 1.
        # To maximize k_x + k_y = (k_y + z + 1) + k_y = 2k_y + z + 1, we maximize k_y.
        # k_y is limited by k_y <= y and k_x <= k_y + z + 1 => k_y >= k_x - z - 1 => y >= k_x - z - 1 => k_y <= x - z - 1 (since k_x <= x).
        # So max k_y = min(y, x - z - 1).
        # Since x > y + z + 1, we have x - z - 1 > y. So min(y, x - z - 1) = y.
        # Max k_y = y. Max k_x = k_y + z + 1 = y + z + 1.
        # Total blocks = k_x + k_y + k_z = (y + z + 1) + y + z = 2y + 2z + 1.
        # This is valid if k_x <= x (y+z+1 <= x ok, as x > y+z+1) and k_y <= y (y <= y ok).

        # The three cases cover all possibilities for x, y, z >= 1.
        # Case 1: x <= y - 1
        # Case 2: x > y + z + 1
        # Case 3 (Else): y - 1 < x <= y + z + 1 (since Case 1 is false)

        num_blocks = 0
        if x <= y - 1:
            num_blocks = 2 * x + 1 + z
        elif x >= y + z + 1:
             num_blocks = 2 * y + 2 * z + 1
        else:
            num_blocks = x + y + z

        return 2 * num_blocks