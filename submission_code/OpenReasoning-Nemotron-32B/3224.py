from typing import List

mod = 10**9 + 7
max_n_val = 100000

class Solution:
	fact = None
	
	def numberOfSequence(self, n: int, sick: List[int]) -> int:
		if Solution.fact is None:
			self.precompute_factorials()
			
		segments = []
		total_mult = 1
		
		left_len = sick[0]
		if left_len > 0:
			segments.append(left_len)
			
		for i in range(len(sick)-1):
			gap = sick[i+1] - sick[i] - 1
			if gap > 0:
				segments.append(gap)
				total_mult = (total_mult * pow(2, gap-1, mod)) % mod
				
		right_len = n - 1 - sick[-1]
		if right_len > 0:
			segments.append(right_len)
			
		total_infections = n - len(sick)
		if total_infections == 0:
			return 1
			
		numerator = Solution.fact[total_infections]
		denominator = 1
		for length in segments:
			denominator = (denominator * Solution.fact[length]) % mod
			
		inv_denom = pow(denominator, mod-2, mod)
		multinomial = numerator * inv_denom % mod
		
		result = multinomial * total_mult % mod
		return result
		
	def precompute_factorials(self):
		n = max_n_val
		Solution.fact = [1] * (n+1)
		for i in range(1, n+1):
			Solution.fact[i] = Solution.fact[i-1] * i % mod