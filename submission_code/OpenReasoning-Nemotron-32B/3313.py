class Solution:
	def maximumStrength(self, nums: List[int], k: int) -> int:
		n = len(nums)
		c = [0] * (k + 1)
		for j in range(1, k + 1):
			if j % 2 == 1:
				c[j] = k - j + 1
			else:
				c[j] = -(k - j + 1)
		
		NEG_INF = -10**18
		dp0_prev = [0] + [NEG_INF] * k
		ep_prev = [NEG_INF] * (k + 1)
		
		ans = NEG_INF
		
		for i in range(n):
			dp0_curr = [NEG_INF] * (k + 1)
			ep_curr = [NEG_INF] * (k + 1)
			for j in range(0, k + 1):
				if j >= 1:
					start_val = dp0_prev[j - 1] + c[j] * nums[i]
					extend_val = ep_prev[j] + c[j] * nums[i]
					ep_curr[j] = max(start_val, extend_val)
				skip_val = dp0_prev[j]
				end_val = ep_curr[j]
				dp0_curr[j] = max(skip_val, end_val)
				if j == k and dp0_curr[j] > ans:
					ans = dp0_curr[j]
			dp0_prev = dp0_curr
			ep_prev = ep_curr
		
		return ans