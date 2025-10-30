class Solution:
	def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
		happiness.sort(reverse=True)
		total = 0
		for i in range(k):
			if happiness[i] > i:
				total += happiness[i] - i
			else:
				break
		return total