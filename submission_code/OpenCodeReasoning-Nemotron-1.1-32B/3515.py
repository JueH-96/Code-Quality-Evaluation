class Solution:
	def canAliceWin(self, nums: List[int]) -> bool:
		total = sum(nums)
		single_digit = [x for x in nums if x < 10]
		double_digit = [x for x in nums if x >= 10]
		sum_single = sum(single_digit)
		sum_double = sum(double_digit)
		return (sum_single > total - sum_single) or (sum_double > total - sum_double)