class Solution:
	def reverseDegree(self, s: str) -> int:
		total = 0
		for i, char in enumerate(s):
			reversed_pos = 26 - (ord(char) - ord('a'))
			total += reversed_pos * (i + 1)
		return total