from collections import defaultdict

class Solution:
	def maximumLength(self, s: str) -> int:
		groups_by_char = defaultdict(list)
		i = 0
		n = len(s)
		while i < n:
			j = i
			while j < n and s[j] == s[i]:
				j += 1
			groups_by_char[s[i]].append(j - i)
			i = j
		
		best = 0
		for char, groups in groups_by_char.items():
			max_len = max(groups)
			if max_len <= best:
				continue
			for k in range(max_len, best, -1):
				total_occurrences = 0
				for L in groups:
					if L >= k:
						total_occurrences += (L - k + 1)
				if total_occurrences >= 3:
					best = k
					break
		
		return best if best > 0 else -1