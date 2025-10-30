from typing import List
from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_num = 0
        # generate all possible orders
        for order in permutations(nums):
            # form the concatenated binary string without leading zeros (bin(x)[2:] gives that)
            bin_str = ''.join(bin(x)[2:] for x in order)
            value = int(bin_str, 2)
            max_num = max(max_num, value)
        return max_num

# Example usage:
# sol = Solution()
# print(sol.maxGoodNumber([1, 2, 3]))  # Output: 30
# print(sol.maxGoodNumber([2, 8, 16])) # Output: 1296