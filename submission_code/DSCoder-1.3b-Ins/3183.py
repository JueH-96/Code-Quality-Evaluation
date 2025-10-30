class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        k_or = 0
        for i in range(len(nums)):
            if nums[i] & (2**i) == (2**i):
                k -= 1
            if k == 0:
                k_or += 2**i
        return k_or