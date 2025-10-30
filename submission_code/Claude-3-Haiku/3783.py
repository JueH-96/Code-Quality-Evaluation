from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        if n < 1 or k < 1:
            return []

        # Generate all possible alternating permutations
        perms = self.generate_alternating_perms(n)

        # Sort the permutations lexicographically
        perms.sort()

        # Return the k-th permutation if it exists, otherwise return an empty list
        return perms[k-1] if k <= len(perms) else []

    def generate_alternating_perms(self, n: int) -> List[List[int]]:
        if n == 1:
            return [[1]]

        perms = []
        for i in range(1, n+1, 2):
            for j in range(2, n+1, 2):
                for perm in self.generate_alternating_perms(n-2):
                    perms.append([i] + perm + [j])
        return perms