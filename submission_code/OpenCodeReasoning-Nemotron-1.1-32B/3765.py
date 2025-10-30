class Solution:
	def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
		n = len(nums)
		S = [0] * (n + 1)
		C_arr = [0] * (n + 1)
		for i in range(1, n + 1):
			S[i] = S[i - 1] + nums[i - 1]
			C_arr[i] = C_arr[i - 1] + cost[i - 1]
		
		INF = 10**18
		best = INF
		dp_prev = [INF] * (n + 1)
		dp_prev[0] = 0
		
		for t_val in range(1, n + 1):
			dp_curr = [INF] * (n + 1)
			for i in range(0, n):
				for j in range(0, i + 1):
					if t_val == 1:
						if j != 0:
							continue
					else:
						if j < t_val - 1:
							continue
					total_cost = dp_prev[j] + (S[i + 1] + k * t_val) * (C_arr[i + 1] - C_arr[j])
					if total_cost < dp_curr[i + 1]:
						dp_curr[i + 1] = total_cost
			if dp_curr[n] < best:
				best = dp_curr[n]
			dp_prev = dp_curr
		
		return best