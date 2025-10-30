class Solution:
	def makeTheIntegerZero(self, num1: int, num2: int) -> int:
		max_val = 1 << 60
		for k in range(0, 61):
			x = num1 - k * num2
			if x < 0:
				continue
			if x < k:
				continue
			if x > k * max_val:
				continue
			ones = bin(x).count('1')
			if k >= ones:
				return k
		return -1