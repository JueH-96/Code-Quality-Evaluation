class Solution:
	def minExtraChar(self, s: str, dictionary: List[str]) -> int:
		word_set = set(dictionary)
		n = len(s)
		dp = [0] * (n + 1)
		for i in range(1, n + 1):
			dp[i] = dp[i - 1] + 1
			for j in range(0, i):
				if s[j:i] in word_set:
					if dp[j] < dp[i]:
						dp[i] = dp[j]
		return dp[n]