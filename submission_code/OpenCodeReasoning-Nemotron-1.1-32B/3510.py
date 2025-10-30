class Solution:
	def maximumTotalSum(self, maximumHeight: List[int]) -> int:
		maximumHeight.sort(reverse=True)
		total = maximumHeight[0]
		current = maximumHeight[0]
		for i in range(1, len(maximumHeight)):
			current = min(maximumHeight[i], current - 1)
			if current <= 0:
				return -1
			total += current
		return total