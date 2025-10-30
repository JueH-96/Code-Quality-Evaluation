import math
from typing import List

class Solution:
	def subsequencePairCount(self, nums: List[int]) -> int:
		mod = 10**9 + 7
		M = 200
		dp = [[0] * (M + 1) for _ in range(M + 1)]
		dp[0][0] = 1
		
		for x in nums:
			new_dp = [[0] * (M + 1) for _ in range(M + 1)]
			for g1 in range(M + 1):
				for g2 in range(M + 1):
					cnt = dp[g1][g2]
					if cnt == 0:
						continue
					new_dp[g1][g2] = (new_dp[g1][g2] + cnt) % mod
					ng1 = math.gcd(g1, x)
					new_dp[ng1][g2] = (new_dp[ng1][g2] + cnt) % mod
					ng2 = math.gcd(g2, x)
					new_dp[g1][ng2] = (new_dp[g1][ng2] + cnt) % mod
			dp = new_dp
		
		ans = 0
		for g in range(1, M + 1):
			ans = (ans + dp[g][g]) % mod
		return ans