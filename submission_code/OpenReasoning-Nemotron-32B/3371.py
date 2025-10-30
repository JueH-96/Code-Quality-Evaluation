class Solution:
	def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
		s = str(x)
		total = sum(int(digit) for digit in s)
		if x % total == 0:
			return total
		else:
			return -1