class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        stack = []
        max_sum = float('-inf')

        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]

        for i in range(n - 1, -1, -1):
            while stack and nums[i] >= nums[stack[-1]]:
                j = stack.pop()
                if stack:
                    k = stack[-1]
                    max_sum = max(max_sum, prefix_sum[j] - prefix_sum[k])
            stack.append(i)

        return max_sum if max_sum != float('-inf') else -1