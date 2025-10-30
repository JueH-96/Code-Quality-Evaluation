from collections import Counter

class Solution:
	def maximumTotalDamage(self, power: List[int]) -> int:
		if not power:
			return 0
		distinct = sorted(set(power))
		freq = Counter(power)
		n = len(distinct)
		weights = [freq[d] * d for d in distinct]
		
		dp = [0] * n
		j = -1
		for i in range(n):
			while j + 1 < i and distinct[j + 1] <= distinct[i] - 3:
				j += 1
			take = weights[i] + (dp[j] if j >= 0 else 0)
			skip = dp[i - 1] if i > 0 else 0
			dp[i] = max(skip, take)
		
		return dp[-1]