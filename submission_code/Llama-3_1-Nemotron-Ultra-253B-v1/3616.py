from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            if nums[i] == 0:
                for direction in [-1, 1]:
                    copy = list(nums)
                    curr = i
                    step = direction
                    while True:
                        if curr < 0 or curr >= n:
                            break
                        if copy[curr] == 0:
                            curr += step
                        else:
                            copy[curr] -= 1
                            step = -step
                            curr += step
                    if all(x == 0 for x in copy):
                        count += 1
        return count