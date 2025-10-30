class Solution:
	def maxSum(self, nums: List[int], k: int, m: int) -> int:
		n = len(nums)
		NEG_INF = -10**18
		dp = [[NEG_INF] * (m+1) for _ in range(k+1)]
		dp[0][0] = 0
		
		for i in range(n):
			for j in range(k):
				if dp[j][m] != NEG_INF:
					if dp[j+1][0] < dp[j][m]:
						dp[j+1][0] = dp[j][m]
			
			dp_next = [[NEG_INF] * (m+1) for _ in range(k+1)]
			
			for j in range(k+1):
				for l in range(m+1):
					if dp[j][l] == NEG_INF:
						continue
					if l == 0:
						if dp_next[j][0] < dp[j][l]:
							dp_next[j][0] = dp[j][l]
						if j < k:
							l_new = 1
							if l_new >= m:
								l_new = m
							new_val = dp[j][l] + nums[i]
							if dp_next[j][l_new] < new_val:
								dp_next[j][l_new] = new_val
					else:
						l_new = l + 1
						if l_new >= m:
							l_new = m
						new_val = dp[j][l] + nums[i]
						if dp_next[j][l_new] < new_val:
							dp_next[j][l_new] = new_val
			dp = dp_next
		
		for j in range(k):
			if dp[j][m] != NEG_INF:
				if dp[j+1][0] < dp[j][m]:
					dp[j+1][0] = dp[j][m]
		
		return dp[k][0]