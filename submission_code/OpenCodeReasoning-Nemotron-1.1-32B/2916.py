class Solution:
	def canSplitArray(self, nums: List[int], m: int) -> bool:
		n = len(nums)
		prefix = [0] * (n + 1)
		for i in range(1, n + 1):
			prefix[i] = prefix[i - 1] + nums[i - 1]
		
		dp = [[False] * n for _ in range(n)]
		for i in range(n):
			dp[i][i] = True
		
		for L in range(2, n + 1):
			for i in range(0, n - L + 1):
				j = i + L - 1
				for k in range(i, j):
					left_len = k - i + 1
					right_len = j - k
					left_sum = prefix[k + 1] - prefix[i]
					right_sum = prefix[j + 1] - prefix[k + 1]
					
					cond_left = (left_len == 1) or (left_sum >= m)
					cond_right = (right_len == 1) or (right_sum >= m)
					
					if cond_left and cond_right:
						if dp[i][k] and dp[k + 1][j]:
							dp[i][j] = True
							break
		return dp[0][n - 1]