class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        if x == y:
            return 2 * x + 2 * z
        elif x > y:
            diff = x - y
            if diff == 1:
                return 2 * x + 2 * y + 2 * z
            else:
                # We can use at most y "BB" and y + 1 "AA"
                len_aa_bb = 2 * (y + 1) + 2 * y
                # We can insert "AB" to separate "AA" sequences
                num_ab = min(z, y + 1)
                return len_aa_bb + 2 * num_ab
        else:  # y > x
            diff = y - x
            if diff == 1:
                return 2 * x + 2 * y + 2 * z
            else:
                # We can use at most x "AA" and x + 1 "BB"
                len_aa_bb = 2 * x + 2 * (x + 1)
                # We can insert "AB" to separate "BB" sequences
                num_ab = min(z, x + 1)
                return len_aa_bb + 2 * num_ab