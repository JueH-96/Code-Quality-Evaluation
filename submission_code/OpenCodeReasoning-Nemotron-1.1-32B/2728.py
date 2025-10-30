class Solution:
	def matrixSum(self, nums: List[List[int]]) -> int:
		sorted_nums = [sorted(row, reverse=True) for row in nums]
		max_cols = max(len(row) for row in sorted_nums)
		total = 0
		for j in range(max_cols):
			total += max(row[j] for row in sorted_nums if j < len(row))
		return total