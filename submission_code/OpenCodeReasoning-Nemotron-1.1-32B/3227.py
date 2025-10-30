class Solution:
	def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
		n = len(grid)
		total = n * n
		total_sum = total * (total + 1) // 2
		actual_sum = 0
		seen = set()
		duplicate = None
		for row in grid:
			for num in row:
				actual_sum += num
				if num in seen:
					duplicate = num
				else:
					seen.add(num)
		missing = total_sum - actual_sum + duplicate
		return [duplicate, missing]