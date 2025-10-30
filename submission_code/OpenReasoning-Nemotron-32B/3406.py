mod = 10**9 + 7

class Solution:
	def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
		dp = [[[[0] * (limit + 1) for _ in range(2)] for _ in range(one + 1)] for _ in range(zero + 1)]
		
		if zero >= 1:
			dp[1][0][0][1] = 1
		if one >= 1:
			dp[0][1][1][1] = 1
			
		n = zero + one
		for total in range(1, n):
			start_z = max(0, total - one)
			end_z = min(zero, total)
			for z in range(start_z, end_z + 1):
				o = total - z
				for last in [0, 1]:
					for run in range(1, limit + 1):
						cnt = dp[z][o][last][run]
						if cnt == 0:
							continue
						if z < zero:
							if last == 0:
								new_run = run + 1
								if new_run <= limit:
									dp[z + 1][o][0][new_run] = (dp[z + 1][o][0][new_run] + cnt) % mod
							else:
								dp[z + 1][o][0][1] = (dp[z + 1][o][0][1] + cnt) % mod
						if o < one:
							if last == 1:
								new_run = run + 1
								if new_run <= limit:
									dp[z][o + 1][1][new_run] = (dp[z][o + 1][1][new_run] + cnt) % mod
							else:
								dp[z][o + 1][1][1] = (dp[z][o + 1][1][1] + cnt) % mod
		
		ans = 0
		for last in [0, 1]:
			for run in range(1, limit + 1):
				ans = (ans + dp[zero][one][last][run]) % mod
		return ans