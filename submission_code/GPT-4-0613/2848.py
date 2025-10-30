from typing import List
from itertools import permutations

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        count = 0
        for perm in permutations(nums):
            for i in range(len(perm) - 1):
                if perm[i] % perm[i+1] != 0 and perm[i+1] % perm[i] != 0:
                    break
            else:
                count += 1
        return count % mod