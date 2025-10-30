import math
from typing import List

class Solution:
	def subsequencePairCount(self, nums: List[int]) -> int:
		mod = 10**9 + 7
		max_val = 200
		
		mu = [1] * (max_val + 1)
		is_prime = [True] * (max_val + 1)
		primes = []
		for i in range(2, max_val + 1):
			if is_prime[i]:
				primes.append(i)
				mu[i] = -1
			for p in primes:
				if i * p > max_val:
					break
				is_prime[i * p] = False
				if i % p == 0:
					mu[i * p] = 0
					break
				else:
					mu[i * p] = -mu[i]
		
		max_exp = 200
		pow2 = [1] * (max_exp + 1)
		pow3 = [1] * (max_exp + 1)
		for i in range(1, max_exp + 1):
			pow2[i] = (pow2[i - 1] * 2) % mod
			pow3[i] = (pow3[i - 1] * 3) % mod
		
		total_ans = 0
		for g in range(1, 201):
			T = []
			for x in nums:
				if x % g == 0:
					T.append(x // g)
			if not T:
				continue
			M = max(T)
			freq = [0] * (M + 1)
			for x in T:
				if x <= M:
					freq[x] += 1
			
			count_div = [0] * (M + 1)
			for d in range(1, M + 1):
				total_count = 0
				j = d
				while j <= M:
					total_count += freq[j]
					j += d
				count_div[d] = total_count
			
			divisors = [d for d in range(1, M + 1) if count_div[d] > 0]
			ans_T = 0
			for d1 in divisors:
				for d2 in divisors:
					g_val = math.gcd(d1, d2)
					lcm_val = d1 * d2 // g_val
					if lcm_val > M:
						c = 0
						a = count_div[d1]
						b = count_div[d2]
					else:
						c = count_div[lcm_val]
						a = count_div[d1] - c
						b = count_div[d2] - c
					
					part1 = pow2[a + b] * pow3[c] % mod
					part2 = pow2[b + c]
					part3 = pow2[a + c]
					F_val = (part1 - part2 - part3 + 1) % mod
					if F_val < 0:
						F_val += mod
					
					term = mu[d1] * mu[d2] * F_val
					ans_T = (ans_T + term) % mod
			
			total_ans = (total_ans + ans_T) % mod
		
		return total_ans