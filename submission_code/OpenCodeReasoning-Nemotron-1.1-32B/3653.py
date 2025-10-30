class Solution:
	def maxSubarraySum(self, nums: List[int], k: int) -> int:
		n = len(nums)
		min_prefix = [10**18] * k
		min_prefix[0] = 0
		current_sum = 0
		max_sum = -10**18
		for i in range(n):
			current_sum += nums[i]
			r = (i + 1) % k
			candidate = current_sum - min_prefix[r]
			if candidate > max_sum:
				max_sum = candidate
			if current_sum < min_prefix[r]:
				min_prefix[r] = current_sum
		return max_sum