class Solution:
	def minTime(self, skill: List[int], mana: List[int]) -> int:
		n = len(skill)
		m = len(mana)
		if m == 0:
			return 0
		dp = [0] * m
		dp[0] = skill[0] * mana[0]
		for j in range(1, m):
			dp[j] = dp[j-1] + skill[0] * mana[j]
		
		for i in range(1, n):
			new_dp = [0] * m
			new_dp[0] = dp[0] + skill[i] * mana[0]
			for j in range(1, m):
				new_dp[j] = max(new_dp[j-1], dp[j]) + skill[i] * mana[j]
			dp = new_dp
		
		return dp[-1]