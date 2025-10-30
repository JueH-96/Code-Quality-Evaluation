from typing import List

class Solution:
	def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
		n = len(receiver)
		MAX_LOG = 35
		dp = [[0] * n for _ in range(MAX_LOG)]
		sum_dp = [[0] * n for _ in range(MAX_LOG)]
		
		for i in range(n):
			dp[0][i] = receiver[i]
			sum_dp[0][i] = receiver[i]
		
		for j in range(1, MAX_LOG):
			for i in range(n):
				mid = dp[j-1][i]
				dp[j][i] = dp[j-1][mid]
				sum_dp[j][i] = sum_dp[j-1][i] + sum_dp[j-1][mid]
		
		ans = 0
		for x in range(n):
			cur = x
			total = x
			for j in range(MAX_LOG):
				if (k >> j) & 1:
					total += sum_dp[j][cur]
					cur = dp[j][cur]
			if total > ans:
				ans = total
		return ans