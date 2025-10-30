import math

def generate_primes(n):
	sieve = [True] * (n + 1)
	sieve[0] = sieve[1] = False
	for i in range(2, int(math.isqrt(n)) + 1):
		if sieve[i]:
			sieve[i*i : n+1 : i] = [False] * len(sieve[i*i : n+1 : i])
	primes = [i for i, is_prime in enumerate(sieve) if is_prime]
	return primes

class Solution:
	def maximumSum(self, nums: List[int]) -> int:
		max_val = 31622
		primes = generate_primes(max_val)
		
		def get_signature(x):
			sig = 1
			for p in primes:
				if p * p > x:
					break
				if x % p == 0:
					count = 0
					while x % p == 0:
						count += 1
						x //= p
					if count % 2 == 1:
						sig *= p
			if x > 1:
				sig *= x
			return sig
		
		group_sums = {}
		for num in nums:
			sig = get_signature(num)
			if sig in group_sums:
				group_sums[sig] += num
			else:
				group_sums[sig] = num
		
		return max(group_sums.values())