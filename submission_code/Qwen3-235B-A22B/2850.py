class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        if x == y:
            sum_aa_bb = x + y
        else:
            sum_aa_bb = 2 * min(x, y) + 1
        return (sum_aa_bb + z) * 2