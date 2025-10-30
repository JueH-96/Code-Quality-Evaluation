class Solution:
	def maxSum(self, nums: List[int]) -> int:
		distinct = set(nums)
		positive_sum = sum(x for x in distinct if x > 0)
		if positive_sum > 0:
			return positive_sum
		else:
			return max(nums)