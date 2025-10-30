class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        for i in range(n - k + 1):
            if prefix[i + k] - prefix[i] < k:
                return False
        return True