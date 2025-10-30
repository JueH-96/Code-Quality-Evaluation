from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            if nums[i] == 0:
                for direction in (-1, 1):
                    arr = list(nums)
                    curr = i
                    d = direction
                    while 0 <= curr < len(arr):
                        if arr[curr] == 0:
                            curr += d
                        else:
                            arr[curr] -= 1
                            d *= -1
                            curr += d
                    if all(x == 0 for x in arr):
                        count += 1
        return count