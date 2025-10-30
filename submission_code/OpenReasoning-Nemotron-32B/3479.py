class Solution:
	def numberOfSubstrings(self, s: str) -> int:
		n = len(s)
		total = 0
		for start in range(n):
			zeros = 0
			ones = 0
			for end in range(start, n):
				if s[end] == '0':
					zeros += 1
				else:
					ones += 1
				if ones >= zeros * zeros:
					total += 1
				if zeros * zeros > (n - start) - zeros:
					break
		return total