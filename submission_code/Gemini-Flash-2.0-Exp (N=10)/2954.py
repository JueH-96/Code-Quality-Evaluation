class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        max_sum = 0
        for i in range(n - k + 1):
            sub_array = nums[i:i+k]
            distinct_count = len(set(sub_array))
            if distinct_count >= m:
                current_sum = sum(sub_array)
                max_sum = max(max_sum, current_sum)
        return max_sum