class Solution:
	def stringHash(self, s: str, k: int) -> str:
		result = []
		n = len(s)
		for i in range(0, n, k):
			total = 0
			for j in range(i, i + k):
				total += ord(s[j]) - ord('a')
			hashed_char = total % 26
			result.append(chr(ord('a') + hashed_char))
		return ''.join(result)