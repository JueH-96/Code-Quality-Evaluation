MOD = 10**9 + 7

class Solution:
	def numberOfWays(self, s: str, t: str, k: int) -> int:
		n = len(s)
		s2 = s + s
		
		if n == 0:
			return 0
		
		pi = [0] * n
		j = 0
		for i in range(1, n):
			while j > 0 and t[i] != t[j]:
				j = pi[j - 1]
			if t[i] == t[j]:
				j += 1
			else:
				j = 0
			pi[i] = j
		
		j = 0
		occurrences = []
		for i in range(len(s2)):
			while j > 0 and s2[i] != t[j]:
				j = pi[j - 1]
			if s2[i] == t[j]:
				j += 1
			else:
				j = 0
			if j == n:
				start_index = i - n + 1
				occurrences.append(start_index)
				j = pi[j - 1]
		
		R = [idx for idx in occurrences if idx < n]
		m0 = 1 if 0 in R else 0
		m1 = len(R) - m0
		
		term1 = pow(n - 1, k, MOD)
		
		if k % 2 == 0:
			sign = 1
		else:
			sign = MOD - 1
		
		inv_n = pow(n, MOD - 2, MOD)
		
		part0 = (term1 + (n - 1) * sign) % MOD
		part0 = (part0 * inv_n) % MOD
		
		part1 = (term1 - sign) % MOD
		part1 = (part1 * inv_n) % MOD
		
		ans = (m0 * part0 + m1 * part1) % MOD
		return ans