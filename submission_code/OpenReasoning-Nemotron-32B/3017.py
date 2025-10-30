class Solution:
	def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
		def count_up_to(n, k):
			if n == 0:
				return 0
			s = str(n)
			len_n = len(s)
			dp = [[[[[0] * (2*len_n+1) for _ in range(k)] for _ in range(2)] for _ in range(2)] for _ in range(len_n+1)]
			dp[0][1][0][0][len_n] = 1
			
			for i in range(len_n):
				for tight in [0,1]:
					for started in [0,1]:
						for mod_val in range(k):
							for balance_val in range(-len_n, len_n+1):
								idx_balance = balance_val + len_n
								if dp[i][tight][started][mod_val][idx_balance] == 0:
									continue
								current_count = dp[i][tight][started][mod_val][idx_balance]
								low_digit = 0
								high_digit = int(s[i]) if tight else 9
								for d in range(low_digit, high_digit+1):
									new_tight = tight and (d == high_digit)
									if started or d > 0:
										new_started = 1
										if d % 2 == 0:
											new_balance = balance_val + 1
										else:
											new_balance = balance_val - 1
									else:
										new_started = started
										new_balance = balance_val
									new_mod = (mod_val * 10 + d) % k
									new_idx_balance = new_balance + len_n
									if 0 <= new_idx_balance < 2*len_n+1:
										dp[i+1][new_tight][new_started][new_mod][new_idx_balance] += current_count
			total = 0
			for tight in [0,1]:
				total += dp[len_n][tight][1][0][len_n]
			return total
		
		return count_up_to(high, k) - count_up_to(low-1, k)