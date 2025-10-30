MOD = 10**9 + 7

class Solution:
	def countBalancedPermutations(self, num: str) -> int:
		n = len(num)
		n_even = (n + 1) // 2
		n_odd = n // 2
		
		total_sum = 0
		freq = [0] * 10
		for d in num:
			digit = int(d)
			total_sum += digit
			freq[digit] += 1
		
		if total_sum % 2 != 0:
			return 0
		
		half = total_sum // 2
		if half > 360:
			return 0
		
		max_val = 100
		fact = [1] * (max_val + 1)
		inv_fact = [1] * (max_val + 1)
		for i in range(1, max_val + 1):
			fact[i] = fact[i - 1] * i % MOD
		
		inv_fact[max_val] = pow(fact[max_val], MOD - 2, MOD)
		for i in range(max_val, 0, -1):
			inv_fact[i - 1] = inv_fact[i] * i % MOD
		
		dp = [[0] * 361 for _ in range(n_even + 1)]
		dp[0][0] = 1
		
		for d in range(10):
			new_dp = [[0] * 361 for _ in range(n_even + 1)]
			for j in range(n_even + 1):
				for k in range(361):
					if dp[j][k] == 0:
						continue
					max_t = min(freq[d], n_even - j)
					for t in range(0, max_t + 1):
						j_new = j + t
						k_new = k + t * d
						if k_new > 360:
							continue
						term = dp[j][k] * inv_fact[t] % MOD
						term = term * inv_fact[freq[d] - t] % MOD
						new_dp[j_new][k_new] = (new_dp[j_new][k_new] + term) % MOD
			dp = new_dp
		
		if half < 0 or half > 360:
			ans = 0
		else:
			ans = dp[n_even][half] * fact[n_even] % MOD * fact[n_odd] % MOD
		return ans