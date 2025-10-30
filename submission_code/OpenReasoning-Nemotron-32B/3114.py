class Solution:
	def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
		n = len(maxHeights)
		ans = 0
		for i in range(n):
			total = maxHeights[i]
			cur = maxHeights[i]
			for j in range(i-1, -1, -1):
				cur = min(maxHeights[j], cur)
				total += cur
			cur = maxHeights[i]
			for j in range(i+1, n):
				cur = min(maxHeights[j], cur)
				total += cur
			if total > ans:
				ans = total
		return ans