MAX = 2**60

class Solution:
	def makeTheIntegerZero(self, num1: int, num2: int) -> int:
		if num2 == 0:
			ceil_val = (num1 + MAX - 1) // MAX
			popcount_val = bin(num1).count("1")
			return max(ceil_val, popcount_val)
		elif num2 > 0:
			denominator = num2 + MAX
			n_low = (num1 + num2 + MAX - 1) // denominator
			n_high = num1 // (num2 + 1)
			if n_low > n_high:
				return -1
			n_max = min(n_high, max(n_low, 31))
			for n in range(n_low, n_max + 1):
				S = num1 - n * num2
				if bin(S).count("1") <= n:
					return n
			return -1
		else:
			abs_num2 = -num2
			denominator = MAX - abs_num2
			n_low = (num1 + denominator - 1) // denominator
			n_max = max(n_low, 1000)
			for n in range(n_low, n_max + 1):
				S = num1 - n * num2
				if bin(S).count("1") <= n:
					return n
			return -1