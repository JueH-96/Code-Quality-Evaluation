class Solution:
	def longestSubsequence(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 0:
			return 0
		dp = [dict() for _ in range(n)]
		M = [[0] * 301 for _ in range(n)]
		
		for i in range(n):
			for j in range(i):
				d_val = abs(nums[i] - nums[j])
				candidate = 2
				if M[j][d_val] > 0:
					candidate = max(candidate, M[j][d_val] + 1)
				
				if d_val in dp[i]:
					if candidate > dp[i][d_val]:
						dp[i][d_val] = candidate
				else:
					dp[i][d_val] = candidate
			
			temp = [0] * 301
			for d0, length_val in dp[i].items():
				if d0 < 301:
					if length_val > temp[d0]:
						temp[d0] = length_val
			M[i][300] = temp[300]
			for x in range(299, -1, -1):
				M[i][x] = max(temp[x], M[i][x+1])
		
		ans = 1
		for i in range(n):
			if dp[i]:
				max_here = max(dp[i].values())
				if max_here > ans:
					ans = max_here
		return ans