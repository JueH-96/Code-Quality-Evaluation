class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        cycle_length = 2 * (n - 1)
        pos = k % cycle_length
        if pos < n:
            return pos
        else:
            return cycle_length - pos