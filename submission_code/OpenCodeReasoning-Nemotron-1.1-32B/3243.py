class Solution:
	def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
		base = 10 ** len(s)
		num_s = int(s)
		if num_s > finish:
			return 0
		
		if start > num_s:
			a_min = (start - num_s + base - 1) // base
		else:
			a_min = 0
		a_max = (finish - num_s) // base
		
		if a_min > a_max:
			return 0
		
		def count_up_to(n, limit_val):
			if n < 0:
				return 0
			if n == 0:
				return 1
			s_n = str(n)
			L = len(s_n)
			total = 1
			for k in range(1, L):
				total += limit_val * ((limit_val + 1) ** (k - 1))
			
			dp = [[0] * 2 for _ in range(L + 1)]
			dp[0][1] = 1
			for i in range(L):
				for tight in [0, 1]:
					if dp[i][tight] == 0:
						continue
					current_digit = int(s_n[i])
					low_bound = 1 if i == 0 else 0
					high_bound = current_digit if tight else 9
					high_bound = min(high_bound, limit_val)
					if low_bound > high_bound:
						continue
					for d in range(low_bound, high_bound + 1):
						new_tight = tight and (d == current_digit)
						dp[i + 1][new_tight] += dp[i][tight]
			total += dp[L][0] + dp[L][1]
			return total
		
		return count_up_to(a_max, limit) - count_up_to(a_min - 1, limit)