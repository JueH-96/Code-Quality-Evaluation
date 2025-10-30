import math
import bisect

class Solution:
	max_n = 32000
	primes = None

	def nonSpecialCount(self, l: int, r: int) -> int:
		if Solution.primes is None:
			Solution.primes = self.generate_primes(Solution.max_n)
		
		total = r - l + 1
		low_bound = math.isqrt(l - 1) + 1
		high_bound = math.isqrt(r)
		
		if low_bound > high_bound:
			return total
		
		start = max(2, low_bound)
		if start > high_bound:
			return total
		
		left_index = bisect.bisect_left(Solution.primes, start)
		right_index = bisect.bisect_right(Solution.primes, high_bound)
		count_special = right_index - left_index
		
		return total - count_special
	
	def generate_primes(self, n):
		sieve = [True] * (n + 1)
		sieve[0] = False
		sieve[1] = False
		sqrt_n = math.isqrt(n)
		for i in range(2, sqrt_n + 1):
			if sieve[i]:
				for j in range(i * i, n + 1, i):
					sieve[j] = False
		primes = [i for i, is_prime in enumerate(sieve) if is_prime]
		return primes