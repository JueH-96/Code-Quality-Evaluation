from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                a = nums[i]
                b = nums[j]
                if self.is_almost_equal(a, b):
                    count += 1
        return count

    def is_almost_equal(self, a: int, b: int) -> bool:
        if a == b:
            return True
        a_str = str(a)
        b_str = str(b)
        # Check all possible swaps in a
        for i in range(len(a_str)):
            for j in range(i + 1, len(a_str)):
                swapped = list(a_str)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                swapped_num = int(''.join(swapped))
                if swapped_num == b:
                    return True
        # Check all possible swaps in b
        for i in range(len(b_str)):
            for j in range(i + 1, len(b_str)):
                swapped = list(b_str)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                swapped_num = int(''.join(swapped))
                if swapped_num == a:
                    return True
        return False