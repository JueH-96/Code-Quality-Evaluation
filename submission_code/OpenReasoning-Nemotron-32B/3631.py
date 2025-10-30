MOD = 10**9 + 7

class Solution:
	def countKReducibleNumbers(self, s: str, k: int) -> int:
		m = len(s)
		max_val = 800
		steps_arr = [0] * (max_val + 1)
		steps_arr[1] = 0
		for i in range(2, max_val + 1):
			pc = bin(i).count('1')
			steps_arr[i] = 1 + steps_arr[pc]
		
		S = set()
		for c in range(1, max_val + 1):
			if steps_arr[c] <= k - 1:
				S.add(c)
		
		dp = [[0] * (m + 1) for _ in range(2)]
		dp[1][0] = 1
		
		for i in range(m):
			new_dp = [[0] * (m + 1) for _ in range(2)]
			for tight in range(2):
				for count_val in range(m + 1):
					if dp[tight][count_val] == 0:
						continue
					current_bound = int(s[i]) if tight else 1
					for d in range(0, current_bound + 1):
						new_tight = tight and (d == current_bound)
						new_count = count_val + (1 if d == 1 else 0)
						if new_count <= m:
							new_dp[new_tight][new_count] = (new_dp[new_tight][new_count] + dp[tight][count_val]) % MOD
			dp = new_dp
		
		total = 0
		for count_val in range(1, m + 1):
			if count_val in S:
				total = (total + dp[0][count_val]) % MOD
		return total