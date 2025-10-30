class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        nums.sort(reverse=True)
        res = nums[0]
        for i in range(1, len(nums)):
            while nums[i] < nums[i-1] and k > 0:
                nums[i] *= 2
                k -= 1
            res |= nums[i]
        return res