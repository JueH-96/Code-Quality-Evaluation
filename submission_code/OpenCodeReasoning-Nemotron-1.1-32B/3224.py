mod = 10**9 + 7

class Solution:
	def numberOfSequence(self, n: int, sick: List[int]) -> int:
		M = n - len(sick)
		if M == 0:
			return 1
		
		segment_lengths = []
		extra = 1
		
		if sick[0] > 0:
			segment_lengths.append(sick[0])
		
		for i in range(len(sick) - 1):
			gap = sick[i+1] - sick[i] - 1
			if gap > 0:
				segment_lengths.append(gap)
				extra = (extra * pow(2, gap - 1, mod)) % mod
		
		if sick[-1] < n - 1:
			segment_lengths.append(n - 1 - sick[-1])
		
		fact = [1] * (M + 1)
		for i in range(1, M + 1):
			fact[i] = fact[i-1] * i % mod
		
		numerator = fact[M]
		denominator = 1
		for s in segment_lengths:
			denominator = (denominator * fact[s]) % mod
		
		inv_denom = pow(denominator, mod - 2, mod)
		multinomial = numerator * inv_denom % mod
		
		total_ways = multinomial * extra % mod
		return total_ways