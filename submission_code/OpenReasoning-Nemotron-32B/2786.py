class Solution:
	def longestSemiRepetitiveSubstring(self, s: str) -> int:
		n = len(s)
		max_length = 0
		for i in range(n):
			count_pairs = 0
			for j in range(i, n):
				if j > i and s[j] == s[j-1]:
					count_pairs += 1
				if count_pairs > 1:
					break
				max_length = max(max_length, j - i + 1)
		return max_length