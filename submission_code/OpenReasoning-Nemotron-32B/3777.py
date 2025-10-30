class Solution:
	def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
		S_bound = 1800
		if k < -S_bound or k > S_bound:
			return -1
		
		n = len(nums)
		offset = S_bound
		dp = [[[-1] * (2 * S_bound + 1) for _ in range(2)] for _ in range(2)]
		dp[0][0][0] = 1
		
		for i in range(n):
			new_dp = [[[-1] * (2 * S_bound + 1) for _ in range(2)] for _ in range(2)]
			for started in range(2):
				for parity in range(2):
					for s_index in range(2 * S_bound + 1):
						if dp[started][parity][s_index] == -1:
							continue
						current_product = dp[started][parity][s_index]
						
						if new_dp[started][parity][s_index] < current_product:
							new_dp[started][parity][s_index] = current_product
							
						if not started:
							new_started = 1
							new_parity = 1
							real_s = s_index - offset
							new_real_s = real_s + nums[i]
							if abs(new_real_s) <= S_bound:
								new_s_index = new_real_s + offset
								new_prod = current_product * nums[i]
								if new_prod <= limit:
									if new_dp[new_started][new_parity][new_s_index] < new_prod:
										new_dp[new_started][new_parity][new_s_index] = new_prod
						else:
							real_s = s_index - offset
							if parity == 0:
								new_real_s = real_s + nums[i]
							else:
								new_real_s = real_s - nums[i]
							if abs(new_real_s) <= S_bound:
								new_s_index = new_real_s + offset
								new_prod = current_product * nums[i]
								if new_prod <= limit:
									new_parity_next = 1 - parity
									if new_dp[started][new_parity_next][new_s_index] < new_prod:
										new_dp[started][new_parity_next][new_s_index] = new_prod
			dp = new_dp
		
		ans = -1
		s_index_target = k + offset
		if 0 <= s_index_target <= 2 * S_bound:
			for started in [1]:
				for parity in range(2):
					if dp[started][parity][s_index_target] > ans:
						ans = dp[started][parity][s_index_target]
		
		return ans if ans != -1 else -1