class Solution:
	def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
		n = len(nums)
		min_sum = -1800
		max_sum = 1800
		offset = -min_sum
		size = max_sum - min_sum + 1
		
		dp = [[-1] * 2 for _ in range(size)]
		
		for num in nums:
			new_dp = [[-1] * 2 for _ in range(size)]
			
			for s in range(size):
				for p in range(2):
					if dp[s][p] != -1:
						if new_dp[s][p] < dp[s][p]:
							new_dp[s][p] = dp[s][p]
			
			for s in range(size):
				for p in range(2):
					if dp[s][p] == -1:
						continue
					actual_sum = s - offset
					if p == 0:
						new_sum_val = actual_sum + num
						new_p = 1
					else:
						new_sum_val = actual_sum - num
						new_p = 0
					
					if new_sum_val < min_sum or new_sum_val > max_sum:
						continue
					new_s = new_sum_val + offset
					new_prod = dp[s][p] * num
					if new_prod > limit:
						continue
					if new_dp[new_s][new_p] < new_prod:
						new_dp[new_s][new_p] = new_prod
			
			new_sum_val = num
			new_p = 1
			new_prod = num
			if new_prod <= limit:
				new_s = new_sum_val + offset
				if new_dp[new_s][new_p] < new_prod:
					new_dp[new_s][new_p] = new_prod
			
			dp = new_dp
		
		s_index = k + offset
		if s_index < 0 or s_index >= size:
			return -1
		ans = -1
		for p in range(2):
			if dp[s_index][p] != -1:
				if dp[s_index][p] > ans:
					ans = dp[s_index][p]
		
		return ans if ans != -1 else -1