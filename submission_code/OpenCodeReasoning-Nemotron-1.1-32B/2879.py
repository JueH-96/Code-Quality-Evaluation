class Solution:
	def minimumChanges(self, s: str, k: int) -> int:
		n = len(s)
		divisors_dict = {}
		for L in range(1, n+1):
			divisors = []
			for d in range(1, L):
				if L % d == 0:
					divisors.append(d)
			divisors_dict[L] = divisors
		
		cost_arr = [[10**9] * (n+1) for _ in range(n)]
		
		for i in range(n):
			for j in range(i+1, n+1):
				L = j - i
				if L == 1:
					cost_arr[i][j] = 10**9
				else:
					min_cost = 10**9
					divisors = divisors_dict[L]
					for d in divisors:
						total_cost = 0
						for group_index in range(d):
							group = []
							idx = i + group_index
							while idx < j:
								group.append(s[idx])
								idx += d
							left = 0
							right = len(group) - 1
							group_cost = 0
							while left < right:
								if group[left] != group[right]:
									group_cost += 1
								left += 1
								right -= 1
							total_cost += group_cost
						if total_cost < min_cost:
							min_cost = total_cost
					cost_arr[i][j] = min_cost
		
		dp = [[10**9] * (k+1) for _ in range(n+1)]
		dp[0][0] = 0
		
		for i in range(1, n+1):
			max_j = min(k, i // 2)
			for j in range(1, max_j+1):
				min_p = 2 * (j-1)
				max_p = i - 2
				if min_p > max_p:
					continue
				for p in range(min_p, max_p+1):
					if dp[p][j-1] < 10**9:
						total_cost = dp[p][j-1] + cost_arr[p][i]
						if total_cost < dp[i][j]:
							dp[i][j] = total_cost
		
		return dp[n][k]