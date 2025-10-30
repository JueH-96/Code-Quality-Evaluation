class Solution:
	def maximumNumberOfStringPairs(self, words: List[str]) -> int:
		available = set(words)
		count = 0
		for word in words:
			if word not in available:
				continue
			rev = word[::-1]
			if rev == word:
				continue
			if rev in available:
				count += 1
				available.remove(word)
				available.remove(rev)
		return count