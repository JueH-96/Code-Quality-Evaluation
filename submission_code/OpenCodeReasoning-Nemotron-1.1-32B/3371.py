class Solution:
	def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
		s = sum(int(digit) for digit in str(x))
		if x % s == 0:
			return s
		else:
			return -1