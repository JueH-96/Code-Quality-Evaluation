class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        prefix[0] = 1
        odd = [0] * (n + 1)
        odd[0] = 1
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + (1 if nums[i - 1] % 2 == 1 else 0)
            odd[i] = odd[i - 1] + (1 if nums[i - 1] % 2 == 1 else 0) * (1 if nums[i - 1] == nums[i - 2] else 0)
        res = 0
        for i in range(1, n + 1):
            res += odd[i] * (odd[i] - 1) // 2
            res += (prefix[i] - prefix[i - odd[i]]) * odd[i]
        return res