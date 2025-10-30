from typing import List

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        arr1, arr2 = [], []
        for i, num in enumerate(nums, start=1):
            if i == 1:
                arr1.append(num)
            elif i == 2:
                arr2.append(num)
            else:
                if arr1[-1] > arr2[-1]:
                    arr1.append(num)
                else:
                    arr2.append(num)
        return arr1 + arr2