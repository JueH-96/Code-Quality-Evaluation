class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        current = 0
        for i in range(n):
            current += diff[i]
            if nums[i] < current:
                return False
            remaining = nums[i] - current
            if remaining > 0:
                if i + k > n:
                    return False
                current += remaining
                if i + k < n:
                    diff[i + k] -= remaining
        return True