mod = 10**9 + 7

class Solution:
	def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
		dp0 = [[[0] * (limit + 1) for _ in range(one + 1)] for __ in range(zero + 1)]
		dp1 = [[[0] * (limit + 1) for _ in range(one + 1)] for __ in range(zero + 1)]
		
		if zero >= 1:
			dp0[1][0][1] = 1
		if one >= 1:
			dp1[0][1][1] = 1
		
		for j in range(zero + 1):
			for k in range(one + 1):
				if (j == 0 and k == 0) or (j == 1 and k == 0) or (j == 0 and k == 1):
					continue
				if j >= 1:
					for r in range(1, limit + 1):
						if r == 1:
							total = 0
							for run in range(1, limit + 1):
								total = (total + dp1[j - 1][k][run]) % mod
							dp0[j][k][r] = total
						else:
							dp0[j][k][r] = dp0[j - 1][k][r - 1]
				if k >= 1:
					for r in range(1, limit + 1):
						if r == 1:
							total = 0
							for run in range(1, limit + 1):
								total = (total + dp0[j][k - 1][run]) % mod
							dp1[j][k][r] = total
						else:
							dp1[j][k][r] = dp1[j][k - 1][r - 1]
		
		total_ways = 0
		for r in range(1, limit + 1):
			total_ways = (total_ways + dp0[zero][one][r]) % mod
			total_ways = (total_ways + dp1[zero][one][r]) % mod
		
		return total_ways