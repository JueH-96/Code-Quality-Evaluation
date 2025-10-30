from typing import List
from itertools import permutations

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        def is_alternating(perm):
            for i in range(len(perm) - 1):
                if (perm[i] % 2 == perm[i + 1] % 2):
                    return False
            return True

        numbers = list(range(1, n + 1))
        alternating_perms = []

        for perm in permutations(numbers):
            if is_alternating(perm):
                alternating_perms.append(list(perm))
                if len(alternating_perms) == k:
                    return alternating_perms[-1]

        return []

# Example usage:
# sol = Solution()
# print(sol.permute(4, 6))  # Output: [3, 4, 1, 2]
# print(sol.permute(3, 2))  # Output: [3, 2, 1]
# print(sol.permute(2, 3))  # Output: []