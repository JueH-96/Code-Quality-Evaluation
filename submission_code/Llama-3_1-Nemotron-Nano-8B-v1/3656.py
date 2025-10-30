from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        for k in range(n + 1):
            s = 0
            for _ in range(k):
                current_length = n - s
                if current_length <= 0:
                    break
                remove = min(3, current_length)
                s += remove
            if s >= n:
                return k
            sub = nums[s:]
            if len(sub) == len(set(sub)):
                return k
        return n