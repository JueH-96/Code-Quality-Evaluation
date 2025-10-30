class Solution:
	def minimumSteps(self, s: str) -> int:
		count_ones = 0
		total_steps = 0
		for char in s:
			if char == '1':
				count_ones += 1
			else:
				total_steps += count_ones
		return total_steps