class Solution:
	def largestPerimeter(self, nums: List[int]) -> int:
		nums.sort()
		total = sum(nums)
		suffix_sum = 0
		n = len(nums)
		for j in range(n-1, 1, -1):
			if total - suffix_sum > 2 * nums[j]:
				return total - suffix_sum
			suffix_sum += nums[j]
		return -1