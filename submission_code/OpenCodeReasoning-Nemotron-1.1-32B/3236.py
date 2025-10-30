class Solution:
	def missingInteger(self, nums: List[int]) -> int:
		total = nums[0]
		n = len(nums)
		for i in range(1, n):
			if nums[i] == nums[i-1] + 1:
				total += nums[i]
			else:
				break
		num_set = set(nums)
		x = total
		while x in num_set:
			x += 1
		return x