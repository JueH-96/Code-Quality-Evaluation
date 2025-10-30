class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        res = 0
        for i in range(31, -1, -1):
            bit = (1 << i)
            count = sum((num & bit) == bit for num in nums)
            if count >= k:
                res |= bit
        return res