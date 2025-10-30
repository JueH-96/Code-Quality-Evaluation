import math

class Solution:
	def nonSpecialCount(self, l: int, r: int) -> int:
		total = r - l + 1
		low_bound = math.isqrt(l - 1) + 1
		high_bound = math.isqrt(r)
		
		if low_bound > high_bound:
			return total
		
		n = high_bound
		if n < 2:
			return total
		
		sieve = [True] * (n + 1)
		sieve[0] = False
		sieve[1] = False
		
		sqrt_n = math.isqrt(n)
		for i in range(2, sqrt_n + 1):
			if sieve[i]:
				for j in range(i * i, n + 1, i):
					sieve[j] = False
		
		count_special = 0
		for num in range(low_bound, high_bound + 1):
			if sieve[num]:
				count_special += 1
		
		return total - count_special