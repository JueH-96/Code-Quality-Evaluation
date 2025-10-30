class Solution:
	def minimumOperations(self, nums: List[int], target: List[int]) -> int:
		n = len(nums)
		total_ops = 0
		prev = 0
		for i in range(n):
			a = target[i] - nums[i]
			if i == 0:
				if a > 0:
					total_ops += a
			else:
				diff = a - prev
				if diff > 0:
					total_ops += diff
			prev = a
		if -prev > 0:
			total_ops += -prev
		return total_ops