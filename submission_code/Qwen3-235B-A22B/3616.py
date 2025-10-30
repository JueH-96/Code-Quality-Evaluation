from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        res = 0
        n = len(nums)
        for i in range(n):
            if nums[i] == 0:
                # Try both directions
                for direction in [1, -1]:
                    copy = list(nums)
                    curr = i
                    dir = direction
                    while True:
                        if curr < 0 or curr >= n:
                            break
                        if copy[curr] == 0:
                            curr += dir
                        else:
                            copy[curr] -= 1
                            dir *= -1
                            curr += dir
                    # Check if all elements are zero
                    if all(x == 0 for x in copy):
                        res += 1
        return res