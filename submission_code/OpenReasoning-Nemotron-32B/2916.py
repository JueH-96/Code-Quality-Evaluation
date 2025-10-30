class Solution:
	def canSplitArray(self, nums: List[int], m: int) -> bool:
		n = len(nums)
		prefix = [0] * (n + 1)
		for i in range(n):
			prefix[i + 1] = prefix[i] + nums[i]
		
		dp = [[False] * (n + 1) for _ in range(n + 1)]
		
		for i in range(n):
			dp[i][i + 1] = True
			if i + 2 <= n:
				dp[i][i + 2] = True
		
		for length in range(3, n + 1):
			for i in range(n - length + 1):
				j = i + length
				total_sum = prefix[j] - prefix[i]
				if total_sum < m:
					dp[i][j] = False
				else:
					found = False
					for k in range(i + 1, j):
						left_ok = (k - i == 1) or (prefix[k] - prefix[i] >= m)
						right_ok = (j - k == 1) or (prefix[j] - prefix[k] >= m)
						if left_ok and right_ok:
							if dp[i][k] and dp[k][j]:
								found = True
								break
					dp[i][j] = found
		
		return dp[0][n]