class Solution:
	def numberOfSubstrings(self, s: str, k: int) -> int:
		n = len(s)
		total_substrings = n * (n + 1) // 2
		l = 0
		freq = [0] * 26
		invalid = 0
		for r in range(n):
			idx = ord(s[r]) - ord('a')
			freq[idx] += 1
			while l <= r and freq[idx] == k:
				idx_left = ord(s[l]) - ord('a')
				freq[idx_left] -= 1
				l += 1
			invalid += (r - l + 1)
		return total_substrings - invalid