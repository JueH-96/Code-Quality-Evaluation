class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        max_sum = 0
        n = len(nums)
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            distinct_elements = set(subarray)
            if len(distinct_elements) >= m:
                current_sum = sum(subarray)
                max_sum = max(max_sum, current_sum)
        return max_sum