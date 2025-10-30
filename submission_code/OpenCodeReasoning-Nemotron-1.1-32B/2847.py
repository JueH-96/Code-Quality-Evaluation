class Solution:
	def maximumNumberOfStringPairs(self, words: List[str]) -> int:
		available = set(words)
		pairs = 0
		for w in words:
			if w in available:
				available.remove(w)
				rev = w[::-1]
				if rev in available:
					available.remove(rev)
					pairs += 1
		return pairs