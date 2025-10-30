class Solution:
	def smallestNumber(self, n: int, t: int) -> int:
		for num in range(n, 1001):
			s = str(num)
			p = 1
			for c in s:
				p *= int(c)
			if p % t == 0:
				return num