from itertools import permutations

class Solution:
	def minimumString(self, a: str, b: str, c: str) -> str:
		def merge(s1, s2):
			if s1 in s2:
				return s2
			if s2 in s1:
				return s1
			n = min(len(s1), len(s2))
			for k in range(n, 0, -1):
				if s1[-k:] == s2[:k]:
					return s1 + s2[k:]
			return s1 + s2
		
		perms = permutations([a, b, c])
		candidates = set()
		for p in perms:
			temp = merge(p[0], p[1])
			candidate = merge(temp, p[2])
			candidates.add(candidate)
		
		candidates = sorted(candidates, key=lambda s: (len(s), s))
		return candidates[0]