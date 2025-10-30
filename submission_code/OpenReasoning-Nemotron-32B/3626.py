class Solution:
	def smallestNumber(self, n: int, t: int) -> int:
		current = n
		while True:
			num = current
			p = 1
			while num:
				num, r = divmod(num, 10)
				p *= r
				if p == 0:
					break
			if p % t == 0:
				return current
			current += 1