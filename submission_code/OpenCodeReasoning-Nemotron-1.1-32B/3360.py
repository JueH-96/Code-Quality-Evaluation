from collections import Counter

class Solution:
	def minimumDeletions(self, word: str, k: int) -> int:
		if not word:
			return 0
		freq_counter = Counter(word)
		freqs = list(freq_counter.values())
		max_freq = max(freqs)
		total_chars = len(word)
		min_deletions = total_chars
		
		for x in range(0, max_freq + 1):
			total_deletions = 0
			for f in freqs:
				if f < x:
					total_deletions += f
				else:
					total_deletions += max(0, f - (x + k))
			if total_deletions < min_deletions:
				min_deletions = total_deletions
				
		return min_deletions