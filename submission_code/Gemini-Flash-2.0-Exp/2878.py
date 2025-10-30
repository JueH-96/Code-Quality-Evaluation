class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * n
        curr = 0
        for i in range(n):
            curr += diff[i]
            if nums[i] < curr:
                return False
            diff_val = nums[i] - curr
            if diff_val > 0:
                if i + k > n:
                    return False
                curr += diff_val
                if i + k < n:
                    diff[i + k] = -diff_val
        return True