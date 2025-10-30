class Solution:
	def countPrefixSuffixPairs(self, words: List[str]) -> int:
		n = len(words)
		count = 0
		for i in range(n):
			for j in range(i+1, n):
				s1 = words[i]
				s2 = words[j]
				if len(s1) > len(s2):
					continue
				if s2.startswith(s1) and s2.endswith(s1):
					count += 1
		return count