mod = 10**9 + 7

class Solution:
	def numberOfWays(self, s: str, t: str, k: int) -> int:
		n_val = len(s)
		s2 = s + s
		pi = [0] * n_val
		k_val = 0
		for i in range(1, n_val):
			while k_val > 0 and t[k_val] != t[i]:
				k_val = pi[k_val - 1]
			if t[k_val] == t[i]:
				k_val += 1
			pi[i] = k_val
		
		q = 0
		count = 0
		found0 = False
		for i in range(len(s2)):
			while q > 0 and t[q] != s2[i]:
				q = pi[q - 1]
			if t[q] == s2[i]:
				q += 1
			if q == n_val:
				start_index = i - n_val + 1
				if start_index < n_val:
					count += 1
					if start_index == 0:
						found0 = True
				q = pi[q - 1]
		
		m = count
		if m == 0:
			return 0
		
		A = pow(n_val - 1, k, mod)
		if k % 2 == 0:
			B = 1
		else:
			B = mod - 1
		
		c_val = 1 if found0 else 0
		total_ways = (m * A + B * (c_val * n_val - m)) % mod
		inv_n = pow(n_val, mod - 2, mod)
		total_ways = total_ways * inv_n % mod
		return total_ways