class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        curr = 0
        for i in range(n):
            curr += diff[i]
            nums[i] -= curr
            if nums[i] < 0:
                return False
            if nums[i] > 0:
                if i + k > n:
                    return False
                diff[i + k] -= nums[i]
                curr += nums[i]
        return True