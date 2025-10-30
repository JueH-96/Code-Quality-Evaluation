from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)

        # Find the index of 1 and n
        index_1 = nums.index(1)
        index_n = nums.index(n)

        # If 1 is not at the start or n is not at the end, we need to move them
        if index_1 != 0:
            # Move 1 to the start
            moves_for_1 = index_1
        else:
            moves_for_1 = 0

        if index_n != n - 1:
            # Move n to the end
            moves_for_n = n - 1 - index_n
        else:
            moves_for_n = 0

        # If index_n is before index_1, we need to adjust the moves
        if index_n < index_1:
            moves_for_n += 1

        return moves_for_1 + moves_for_n