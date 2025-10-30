from collections import defaultdict

class Solution:
	def maximumLength(self, s: str) -> int:
		n = len(s)
		runs_dict = defaultdict(list)
		i = 0
		while i < n:
			j = i
			while j < n and s[j] == s[i]:
				j += 1
			length = j - i
			runs_dict[s[i]].append(length)
			i = j
		
		best = 0
		for runs in runs_dict.values():
			max_run_c = max(runs)
			max_k_c = 0
			for k in range(1, max_run_c + 1):
				total_occurrences = 0
				for L in runs:
					if L >= k:
						total_occurrences += (L - k + 1)
				if total_occurrences >= 3:
					max_k_c = k
				else:
					break
			if max_k_c > best:
				best = max_k_c
				
		return best if best > 0 else -1