class Solution:
	def maximumOddBinaryNumber(self, s: str) -> str:
		count1 = s.count('1')
		count0 = len(s) - count1
		return '1' * (count1 - 1) + '0' * count0 + '1'