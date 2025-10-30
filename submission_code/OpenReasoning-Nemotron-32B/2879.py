class Solution:
	def minimumChanges(self, s: str, k: int) -> int:
		n = len(s)
		divisors = {}
		for L in range(2, n+1):
			divisors[L] = []
			for d in range(1, L):
				if L % d == 0:
					divisors[L].append(d)
		
		cost_arr = [[10**9] * (n+1) for _ in range(n+1)]
		
		for i in range(n):
			for j in range(i+2, n+1):
				L = j - i
				best = 10**9
				for d in divisors[L]:
					total_cost = 0
					for r in range(d):
						k_val = (L - 1 - r) // d
						left_ptr = i + r
						right_ptr = i + r + d * k_val
						while left_ptr < right_ptr:
							if s[left_ptr] != s[right_ptr]:
								total_cost += 1
							left_ptr += d
							right_ptr -= d
					if total_cost < best:
						best = total_cost
				cost_arr[i][j] = best
		
		dp = [[10**9] * (k+1) for _ in range(n+1)]
		dp[0][0] = 0
		
		for i in range(1, n+1):
			max_t = min(i // 2, k)
			for t in range(1, max_t+1):
				start_j = 2 * (t - 1)
				for j in range(start_j, i-1):
					if dp[j][t-1] != 10**9:
						total_cost_here = dp[j][t-1] + cost_arr[j][i]
						if total_cost_here < dp[i][t]:
							dp[i][t] = total_cost_here
		
		return dp[n][k]