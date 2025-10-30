class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        
        total = 0
        for i in range(n):
            start_index = max(0, i - nums[i])
            total += pre[i + 1] - pre[start_index]
        return total