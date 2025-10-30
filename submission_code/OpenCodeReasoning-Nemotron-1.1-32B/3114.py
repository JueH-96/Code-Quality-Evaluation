class Solution:
	def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
		n = len(maxHeights)
		best = 0
		for i in range(n):
			total = maxHeights[i]
			current = maxHeights[i]
			for j in range(i-1, -1, -1):
				current = min(maxHeights[j], current)
				total += current
			current = maxHeights[i]
			for j in range(i+1, n):
				current = min(maxHeights[j], current)
				total += current
			if total > best:
				best = total
		return best