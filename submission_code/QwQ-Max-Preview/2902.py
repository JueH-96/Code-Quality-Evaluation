class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_digits = [max(int(c) for c in str(num)) for num in nums]
        max_sum = -1
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if max_digits[i] == max_digits[j]:
                    current_sum = nums[i] + nums[j]
                    if current_sum > max_sum:
                        max_sum = current_sum
        return max_sum