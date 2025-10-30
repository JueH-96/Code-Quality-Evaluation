class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n+1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
        max_sum = float('-inf')
        for i in range(k):
            for j in range(i+k, n+1):
                curr_sum = prefix_sum[j] - prefix_sum[i] + nums[i]
                max_sum = max(max_sum, curr_sum)
        return max_sum % (10**9 + 7)