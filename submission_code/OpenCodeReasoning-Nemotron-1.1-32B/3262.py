class Solution:
	def largestPerimeter(self, nums: List[int]) -> int:
		nums.sort()
		total = sum(nums)
		n = len(nums)
		for i in range(n-1, 1, -1):
			if nums[i] < total - nums[i]:
				return total
			total -= nums[i]
		return -1