import math
from typing import List

class Solution:
	def maximumSum(self, nums: List[int]) -> int:
		max_factor = 31622
		sieve_arr = [True] * (max_factor + 1)
		sieve_arr[0] = sieve_arr[1] = False
		for i in range(2, int(math.isqrt(max_factor)) + 1):
			if sieve_arr[i]:
				for j in range(i * i, max_factor + 1, i):
					sieve_arr[j] = False
		primes = [i for i in range(2, max_factor + 1) if sieve_arr[i]]
		
		groups = {}
		for x in nums:
			temp = x
			reduced = 1
			for p in primes:
				if p * p > temp:
					break
				if temp % p == 0:
					count = 0
					while temp % p == 0:
						count += 1
						temp //= p
					if count % 2 == 1:
						reduced *= p
			if temp > 1:
				reduced *= temp
			groups[reduced] = groups.get(reduced, 0) + x
		
		return max(groups.values())