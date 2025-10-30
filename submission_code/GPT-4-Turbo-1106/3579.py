from typing import List
from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_num = 0
        for perm in permutations(nums):
            binary_str = ''.join(bin(x)[2:] for x in perm)
            max_num = max(max_num, int(binary_str, 2))
        return max_num

# Example usage:
# sol = Solution()
# print(sol.maxGoodNumber([1,2,3]))  # Output: 30
# print(sol.maxGoodNumber([2,8,16])) # Output: 1296