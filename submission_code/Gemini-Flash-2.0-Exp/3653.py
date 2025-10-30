class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = float('-inf')
        for i in range(n):
            for j in range(i, n):
                length = j - i + 1
                if length % k == 0:
                    current_sum = sum(nums[i:j+1])
                    max_sum = max(max_sum, current_sum)
        return max_sum