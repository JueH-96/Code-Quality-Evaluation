class Solution:
	def countKReducibleNumbers(self, s: str, k: int) -> int:
		mod = 10**9 + 7
		L = len(s)
		max_n = 800
		min_steps = [0] * (max_n + 1)
		min_steps[1] = 0
		for i in range(2, max_n + 1):
			pc = bin(i).count('1')
			min_steps[i] = 1 + min_steps[pc]
		
		dp = [[0] * (L + 1) for _ in range(2)]
		dp[1][0] = 1
		
		for i in range(L):
			new_dp = [[0] * (L + 1) for _ in range(2)]
			for tight in [0, 1]:
				for cnt in range(L + 1):
					ways = dp[tight][cnt]
					if ways == 0:
						continue
					current_digit = s[i]
					for bit in [0, 1]:
						if tight and bit > int(current_digit):
							continue
						new_tight = tight and (bit == int(current_digit))
						new_cnt = cnt + bit
						new_dp[new_tight][new_cnt] = (new_dp[new_tight][new_cnt] + ways) % mod
			dp = new_dp
		
		total = 0
		for tight in [0, 1]:
			for cnt in range(1, L + 1):
				if tight:
					continue
				if cnt == 1:
					total = (total + dp[tight][cnt]) % mod
				else:
					if 1 + min_steps[cnt] <= k:
						total = (total + dp[tight][cnt]) % mod
		return total