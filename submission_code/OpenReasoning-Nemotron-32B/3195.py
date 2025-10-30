class Solution:
	def minimumSteps(self, s: str) -> int:
		n = len(s)
		total_ones = s.count('1')
		count = 0
		moves = 0
		for i, char in enumerate(s):
			if char == '1':
				moves += (n - total_ones + count - i)
				count += 1
		return moves