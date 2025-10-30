class Solution:
	def paintWalls(self, cost: List[int], time: List[int]) -> int:
		n = len(cost)
		max_s_total = n * 501
		INF = 10**18
		dp = [INF] * (max_s_total + 1)
		dp[0] = 0
		current_max = 0
		
		for i in range(n):
			t_i = time[i]
			c_i = cost[i]
			new_current_max = current_max + t_i + 1
			if new_current_max > max_s_total:
				new_current_max = max_s_total
			for s_val in range(current_max, -1, -1):
				if dp[s_val] == INF:
					continue
				new_s = s_val + t_i + 1
				if new_s > max_s_total:
					continue
				if dp[new_s] > dp[s_val] + c_i:
					dp[new_s] = dp[s_val] + c_i
			current_max = new_current_max
		
		ans = INF
		for s_val in range(n, max_s_total + 1):
			if dp[s_val] < ans:
				ans = dp[s_val]
		return ans