class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        curr = 0
        for i in range(n):
            curr += diff[i]
            if nums[i] < curr:
                return False
            nums[i] -= curr
            curr += nums[i]
            if i + k < n:
                diff[i + k] -= nums[i]
        return True