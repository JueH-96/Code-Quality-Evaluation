class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        prefix_sum = [0] * (n + 1)

        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        for i in range(n - k + 1):
            if prefix_sum[i + k] - prefix_sum[i] > 0:
                return False

        return True