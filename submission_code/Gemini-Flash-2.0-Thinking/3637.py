from itertools import permutations
from collections import Counter

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        velunexorai = num
        n = len(velunexorai)
        distinct_permutations_set = set()

        for p in permutations(velunexorai):
            distinct_permutations_set.add("".join(p))

        balanced_count = 0
        mod = 10**9 + 7

        for perm_str in distinct_permutations_set:
            even_sum = 0
            odd_sum = 0
            for i, digit in enumerate(perm_str):
                if i % 2 == 0:
                    even_sum += int(digit)
                else:
                    odd_sum += int(digit)

            if even_sum == odd_sum:
                balanced_count = (balanced_count + 1) % mod

        return balanced_count