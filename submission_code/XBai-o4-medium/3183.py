from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        res = 0
        for i in range(31):  # Check each bit from 0 to 30 (since nums[i] < 2^31)
            cnt = 0
            for num in nums:
                if num & (1 << i):
                    cnt += 1
            if cnt >= k:
                res |= (1 << i)
        return res