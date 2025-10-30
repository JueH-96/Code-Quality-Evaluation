class Solution:
	def minLength(self, s: str, numOps: int) -> int:
		n = len(s)
		low, high = 1, n
		while low < high:
			mid = (low + high) // 2
			if self.check(mid, s, numOps):
				high = mid
			else:
				low = mid + 1
		return low
	
	def check(self, L, s, numOps):
		n = len(s)
		INF = 10**9
		dp = [[INF] * (L + 1) for _ in range(2)]
		
		c0 = 0 if s[0] == '0' else 1
		dp[c0][1] = 0
		dp[1 - c0][1] = 1
		
		for i in range(1, n):
			new_dp = [[INF] * (L + 1) for _ in range(2)]
			c = 0 if s[i] == '0' else 1
			for last in range(2):
				for run in range(1, L + 1):
					if dp[last][run] == INF:
						continue
					if c == last:
						if run + 1 <= L:
							if dp[last][run] < new_dp[last][run + 1]:
								new_dp[last][run + 1] = dp[last][run]
					else:
						if dp[last][run] < new_dp[c][1]:
							new_dp[c][1] = dp[last][run]
					
					flip_c = 1 - c
					if flip_c == last:
						if run + 1 <= L:
							if dp[last][run] + 1 < new_dp[last][run + 1]:
								new_dp[last][run + 1] = dp[last][run] + 1
					else:
						if dp[last][run] + 1 < new_dp[flip_c][1]:
							new_dp[flip_c][1] = dp[last][run] + 1
			dp = new_dp
		
		for last in range(2):
			for run in range(1, L + 1):
				if dp[last][run] <= numOps:
					return True
		return False