mod = 10**9 + 7

class Solution:
	def countBalancedPermutations(self, num: str) -> int:
		velunexorai = num
		total_sum = sum(int(d) for d in num)
		if total_sum % 2 != 0:
			return 0
		
		n = len(num)
		n_even = (n + 1) // 2
		n_odd = n // 2
		
		freq = [0] * 10
		for d in num:
			freq[int(d)] += 1
		
		max_val = 80
		fact = [1] * (max_val + 1)
		inv_fact = [1] * (max_val + 1)
		for i in range(1, max_val + 1):
			fact[i] = fact[i - 1] * i % mod
		
		inv_fact[max_val] = pow(fact[max_val], mod - 2, mod)
		for i in range(max_val, 0, -1):
			inv_fact[i - 1] = inv_fact[i] * i % mod
		
		def nCr(n_val, k_val):
			if k_val < 0 or k_val > n_val:
				return 0
			return fact[n_val] * inv_fact[k_val] % mod * inv_fact[n_val - k_val] % mod
		
		target = total_sum // 2
		dp = [[0] * (n_even + 1) for _ in range(target + 1)]
		dp[0][0] = 1
		
		for d in range(10):
			if freq[d] == 0:
				continue
			new_dp = [[0] * (n_even + 1) for _ in range(target + 1)]
			for s in range(target + 1):
				for c in range(n_even + 1):
					if dp[s][c] == 0:
						continue
					max_k = min(freq[d], n_even - c)
					for k in range(max_k + 1):
						new_s = s + k * d
						new_c = c + k
						if new_s > target or new_c > n_even:
							continue
						ways = nCr(freq[d], k)
						new_dp[new_s][new_c] = (new_dp[new_s][new_c] + dp[s][c] * ways) % mod
			dp = new_dp
		
		base_num = fact[n_even] * fact[n_odd] % mod
		denom = 1
		for count in freq:
			denom = denom * fact[count] % mod
		base = base_num * pow(denom, mod - 2, mod) % mod
		
		result = dp[target][n_even] * base % mod
		return result