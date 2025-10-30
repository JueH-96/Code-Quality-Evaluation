class Solution:
	def maxStrength(self, nums: List[int]) -> int:
		n = len(nums)
		max_prod = -10**18
		for bitmask in range(1, 1 << n):
			prod = 1
			for i in range(n):
				if bitmask & (1 << i):
					prod *= nums[i]
			if prod > max_prod:
				max_prod = prod
		return max_prod