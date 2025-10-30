from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        arr = list(nums)
        for _ in range(k):
            current_min = min(arr)
            for i in range(len(arr)):
                if arr[i] == current_min:
                    arr[i] *= multiplier
                    break
        return arr