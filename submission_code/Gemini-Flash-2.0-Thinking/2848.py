from typing import List
from itertools import permutations

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        mod = 10**9 + 7

        for perm in permutations(nums):
            is_special = True
            for i in range(n - 1):
                if perm[i] % perm[i+1] != 0 and perm[i+1] % perm[i] != 0:
                    is_special = False
                    break
            if is_special:
                count = (count + 1) % mod
        return count