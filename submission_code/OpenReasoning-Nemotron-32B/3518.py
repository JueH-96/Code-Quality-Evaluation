class Solution:
	def maxScore(self, a: List[int], b: List[int]) -> int:
		dp = [0] + [-10**18] * 4
		for num in b:
			for k in range(4, 0, -1):
				if dp[k-1] != -10**18:
					candidate = dp[k-1] + a[k-1] * num
					if candidate > dp[k]:
						dp[k] = candidate
		return dp[4]