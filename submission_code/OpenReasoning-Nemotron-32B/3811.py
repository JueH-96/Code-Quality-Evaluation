class Solution:
	def reverseDegree(self, s: str) -> int:
		total = 0
		for i, char in enumerate(s):
			reversed_value = 26 - (ord(char) - ord('a'))
			total += reversed_value * (i + 1)
		return total