mod = 10**9 + 7

class Solution:
	def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
		total_sum = 20000
		dp = [0] * (total_sum + 1)
		dp[0] = 1
		
		from collections import Counter
		freq = Counter(nums)
		
		for x, cnt in freq.items():
			if x == 0:
				for j in range(total_sum + 1):
					dp[j] = (dp[j] * (cnt + 1)) % mod
			else:
				for j in range(x, total_sum + 1):
					dp[j] = (dp[j] + dp[j - x]) % mod
				start = (cnt + 1) * x
				if start <= total_sum:
					for j in range(total_sum, start - 1, -1):
						dp[j] = (dp[j] - dp[j - start]) % mod
						if dp[j] < 0:
							dp[j] += mod
		
		total_ways = 0
		for s in range(l, r + 1):
			total_ways = (total_ways + dp[s]) % mod
		return total_ways