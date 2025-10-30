from collections import Counter

class Solution:
	def maxDifference(self, s: str) -> int:
		freq = Counter(s)
		odd_freqs = []
		even_freqs = []
		for count in freq.values():
			if count % 2 == 0:
				even_freqs.append(count)
			else:
				odd_freqs.append(count)
		return max(odd_freqs) - min(even_freqs)