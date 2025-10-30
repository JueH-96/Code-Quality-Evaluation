from collections import defaultdict

class Solution:
	def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
		n = len(word)
		m = n // k
		freq = defaultdict(int)
		for i in range(0, n, k):
			block = word[i:i+k]
			freq[block] += 1
		max_count = max(freq.values())
		return m - max_count