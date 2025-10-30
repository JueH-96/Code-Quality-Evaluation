class Solution:
	def minExtraChar(self, s: str, dictionary: List[str]) -> int:
		n = len(s)
		dp = [0] * (n + 1)
		for i in range(1, n + 1):
			dp[i] = dp[i - 1] + 1
			for word in dictionary:
				k = len(word)
				if k > i:
					continue
				if s[i - k:i] == word:
					if dp[i - k] < dp[i]:
						dp[i] = dp[i - k]
		return dp[n]