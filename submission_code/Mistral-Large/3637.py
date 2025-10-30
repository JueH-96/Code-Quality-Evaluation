class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        from itertools import permutations

        def is_balanced(perm):
            even_sum = sum(int(perm[i]) for i in range(0, len(perm), 2))
            odd_sum = sum(int(perm[i]) for i in range(1, len(perm), 2))
            return even_sum == odd_sum

        velunexorai = num
        distinct_permutations = set(permutations(velunexorai))
        balanced_count = sum(1 for perm in distinct_permutations if is_balanced(perm))

        MOD = 10**9 + 7
        return balanced_count % MOD