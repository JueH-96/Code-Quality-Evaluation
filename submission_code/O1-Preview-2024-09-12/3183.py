class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        res = 0
        MAX_BIT = 31  # Since nums[i] < 2^31
        for i in range(MAX_BIT):
            count = sum(1 for num in nums if num & (1 << i))
            if count >= k:
                res |= (1 << i)
        return res