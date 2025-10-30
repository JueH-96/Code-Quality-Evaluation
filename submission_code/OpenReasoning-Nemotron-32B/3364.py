INF = 10**18

class Solution:
	def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
		n = len(nums)
		m_val = len(andValues)
		if m_val == 0:
			return 0
		dp = [[INF] * n for _ in range(m_val)]
		
		for j in range(m_val):
			for i in range(j, n):
				and_val = (1 << 60) - 1
				for k in range(i, -1, -1):
					and_val &= nums[k]
					if and_val == 0 and andValues[j] != 0:
						break
					if and_val == andValues[j]:
						if j == 0:
							if k == 0:
								candidate = nums[i]
							else:
								candidate = INF
						else:
							if k == 0:
								candidate = INF
							else:
								candidate = dp[j-1][k-1] + nums[i]
						if candidate < dp[j][i]:
							dp[j][i] = candidate
		return dp[m_val-1][n-1] if dp[m_val-1][n-1] < INF else -1