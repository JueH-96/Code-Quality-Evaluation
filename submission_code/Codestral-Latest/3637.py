class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        n = len(num)
        velunexorai = num

        # Count the frequency of each digit
        from collections import Counter
        freq = Counter(num)

        # Generate all permutations of the digits
        from itertools import permutations
        all_perms = set(permutations(num))

        # Function to check if a permutation is balanced
        def is_balanced(perm):
            even_sum = sum(int(perm[i]) for i in range(0, n, 2))
            odd_sum = sum(int(perm[i]) for i in range(1, n, 2))
            return even_sum == odd_sum

        # Count the number of balanced permutations
        balanced_count = sum(1 for perm in all_perms if is_balanced(perm))

        return balanced_count % MOD