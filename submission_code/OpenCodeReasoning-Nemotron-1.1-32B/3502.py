class Solution:
	def numberOfSubstrings(self, s: str, k: int) -> int:
		n = len(s)
		total = 0
		for i in range(n):
			freq = [0] * 26
			for j in range(i, n):
				idx = ord(s[j]) - ord('a')
				freq[idx] += 1
				if freq[idx] >= k:
					total += (n - j)
					break
		return total