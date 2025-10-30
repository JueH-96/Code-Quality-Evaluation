class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # Compute S: total score of all levels
        num_ones = sum(possible)
        num_zeros = n - num_ones
        S = num_ones - num_zeros
        # Compute prefix sums
        prefix = 0
        # We will compute prefix on the fly
        # Iterate through k from 1 to n-1
        for k in range(1, n):
            prefix += 1 if possible[k-1] == 1 else -1
            if 2 * prefix > S:
                return k
        return -1