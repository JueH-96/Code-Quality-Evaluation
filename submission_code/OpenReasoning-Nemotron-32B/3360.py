class Solution:
	def minimumDeletions(self, word: str, k: int) -> int:
		from collections import Counter
		n = len(word)
		if n == 0:
			return 0
		freq_map = Counter(word)
		freqs = list(freq_map.values())
		max_freq = max(freqs)
		min_del = n
		
		for f in range(0, max_freq + 1):
			total = 0
			for count in freqs:
				if count < f:
					total += count
				else:
					total += min(count, max(0, count - (f + k)))
			if total < min_del:
				min_del = total
				
		return min_del