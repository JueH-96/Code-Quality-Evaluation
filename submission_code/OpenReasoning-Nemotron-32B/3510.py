class Solution:
	def maximumTotalSum(self, maximumHeight: List[int]) -> int:
		current = 10**10
		total = 0
		for h in sorted(maximumHeight, reverse=True):
			assign = min(h, current - 1)
			if assign <= 0:
				return -1
			total += assign
			current = assign
		return total