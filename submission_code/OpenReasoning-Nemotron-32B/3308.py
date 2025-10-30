from collections import defaultdict

class Solution:
	def lastNonEmptyString(self, s: str) -> str:
		n = len(s)
		if n == 0:
			return ""
		
		pos = defaultdict(list)
		for idx, char in enumerate(s):
			pos[char].append(idx)
		
		max_freq = 0
		for arr in pos.values():
			if len(arr) > max_freq:
				max_freq = len(arr)
		
		keep = [False] * n
		for arr in pos.values():
			if len(arr) > max_freq - 1:
				for idx in arr[max_freq-1:]:
					keep[idx] = True
		
		result = ''.join(char for idx, char in enumerate(s) if keep[idx])
		return result