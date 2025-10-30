from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        prefix_ones = [0] * (n + 1)
        for i in range(n):
            prefix_ones[i + 1] = prefix_ones[i] + possible[i]
        total_ones = prefix_ones[-1]
        for k in range(1, n):
            a_ones = prefix_ones[k]
            alice_score = 2 * a_ones - k
            bob_ones = total_ones - a_ones
            bob_score = 2 * bob_ones - (n - k)
            if alice_score > bob_score:
                return k
        return -1